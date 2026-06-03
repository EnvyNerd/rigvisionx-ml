package com.rigvisionx.aeos;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class TrainingPanel extends JPanel {
    private final JComboBox<String> trainingType;
    private final JCheckBox saveModels;
    private final JButton attachButton;
    private final JButton trainButton;
    private final JLabel fileLabel;
    private final JLabel statusLabel;
    private final JTextArea outputArea;

    private File selectedFile;
    private final TrainingClient client;

    public TrainingPanel(String baseUrl) {
        this.client = new TrainingClient(baseUrl);

        setLayout(new BorderLayout(12, 12));

        JPanel controls = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(6, 6, 6, 6);

        trainingType = new JComboBox<>(new String[]{"failure_risk", "rul", "anomaly", "all"});
        saveModels = new JCheckBox("Save models on server", true);
        attachButton = new JButton("Attach CSV");
        trainButton = new JButton("Train");
        fileLabel = new JLabel("No file selected");
        statusLabel = new JLabel("Idle");
        outputArea = new JTextArea(10, 40);
        outputArea.setEditable(false);

        gbc.gridx = 0;
        gbc.gridy = 0;
        controls.add(new JLabel("Training type"), gbc);
        gbc.gridx = 1;
        controls.add(trainingType, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        controls.add(new JLabel("Dataset"), gbc);
        gbc.gridx = 1;
        controls.add(fileLabel, gbc);

        gbc.gridx = 2;
        controls.add(attachButton, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        controls.add(saveModels, gbc);

        gbc.gridx = 2;
        controls.add(trainButton, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.gridwidth = 3;
        controls.add(statusLabel, gbc);

        add(controls, BorderLayout.NORTH);
        add(new JScrollPane(outputArea), BorderLayout.CENTER);

        attachButton.addActionListener(evt -> chooseFile());
        trainButton.addActionListener(evt -> startTraining());
    }

    private void chooseFile() {
        JFileChooser chooser = new JFileChooser();
        int result = chooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFile = chooser.getSelectedFile();
            fileLabel.setText(selectedFile.getName());
        }
    }

    private void startTraining() {
        if (selectedFile == null) {
            JOptionPane.showMessageDialog(this, "Attach a CSV first.");
            return;
        }

        trainButton.setEnabled(false);
        statusLabel.setText("Training started...");
        outputArea.setText("");

        SwingWorker<String, Void> worker = new SwingWorker<>() {
            @Override
            protected String doInBackground() throws Exception {
                return client.train(selectedFile.toPath(),
                    (String) trainingType.getSelectedItem(),
                    saveModels.isSelected());
            }

            @Override
            protected void done() {
                try {
                    String response = get();
                    statusLabel.setText("Training complete");
                    outputArea.setText(response);
                } catch (Exception ex) {
                    statusLabel.setText("Training failed");
                    outputArea.setText(ex.getMessage());
                } finally {
                    trainButton.setEnabled(true);
                }
            }
        };

        worker.execute();
    }

    public static class TrainingClient {
        private final HttpClient httpClient;
        private final URI baseUri;

        public TrainingClient(String baseUrl) {
            this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
            this.baseUri = URI.create(baseUrl);
        }

        public String train(Path csvPath, String trainingType, boolean saveModels)
            throws IOException, InterruptedException {
            String boundary = "----TerraEnergyAIBoundary" + System.currentTimeMillis();

            Map<String, String> fields = new LinkedHashMap<>();
            fields.put("training_type", trainingType);
            fields.put("save_models", Boolean.toString(saveModels));

            HttpRequest.BodyPublisher body = buildMultipart(boundary, fields, csvPath);

            HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("/train"))
                .timeout(Duration.ofMinutes(10))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(body)
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                throw new IOException("HTTP " + response.statusCode() + ": " + response.body());
            }

            return response.body();
        }

        private static HttpRequest.BodyPublisher buildMultipart(
            String boundary,
            Map<String, String> fields,
            Path csvPath
        ) throws IOException {
            List<byte[]> byteArrays = new ArrayList<>();

            for (Map.Entry<String, String> entry : fields.entrySet()) {
                byteArrays.add(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
                byteArrays.add(("Content-Disposition: form-data; name=\"" + entry.getKey() + "\"\r\n\r\n")
                    .getBytes(StandardCharsets.UTF_8));
                byteArrays.add(entry.getValue().getBytes(StandardCharsets.UTF_8));
                byteArrays.add("\r\n".getBytes(StandardCharsets.UTF_8));
            }

            if (csvPath != null) {
                String filename = csvPath.getFileName().toString();
                byteArrays.add(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
                byteArrays.add(("Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n")
                    .getBytes(StandardCharsets.UTF_8));
                byteArrays.add("Content-Type: text/csv\r\n\r\n".getBytes(StandardCharsets.UTF_8));
                byteArrays.add(Files.readAllBytes(csvPath));
                byteArrays.add("\r\n".getBytes(StandardCharsets.UTF_8));
            }

            byteArrays.add(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
            return HttpRequest.BodyPublishers.ofByteArrays(byteArrays);
        }
    }
}
