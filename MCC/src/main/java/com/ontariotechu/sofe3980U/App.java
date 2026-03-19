package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;

public class App {
    public static void main(String[] args) {
        String filePath = "model.csv";
        try {
            FileReader filereader = new FileReader(filePath);
            CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
            List<String[]> allData = csvReader.readAll();

            float ce = 0;
            int n = allData.size();
            int[][] confusionMatrix = new int[5][5];

            for (String[] row : allData) {
                int y_true = Integer.parseInt(row[0]);
                float[] y_predicted = new float[5];
                int y_hat = 1;
                float max_prob = -1;

                for (int i = 0; i < 5; i++) {
                    y_predicted[i] = Float.parseFloat(row[i + 1]);
                    if (y_predicted[i] > max_prob) {
                        max_prob = y_predicted[i];
                        y_hat = i + 1; // Class values are 1-5
                    }
                }

                // Cross Entropy
                ce += Math.log(y_predicted[y_true - 1]);
                
                // Populate the matrix: Rows = y^ (predicted), Cols = y (actual)
                confusionMatrix[y_hat - 1][y_true - 1]++;
            }

            ce = -ce / n;

            System.out.println("CE =" + ce);
            System.out.println("Confusion matrix");
            System.out.println("\t\ty=1\ty=2\ty=3\ty=4\ty=5");
            for (int i = 0; i < 5; i++) {
                System.out.print("\ty^=" + (i + 1));
                for (int j = 0; j < 5; j++) {
                    System.out.print("\t" + confusionMatrix[i][j]);
                }
                System.out.println();
            }

        } catch (Exception e) {
            System.out.println("Error reading the CSV file");
        }
    }
}