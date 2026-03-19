package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;

public class App {
    public static void main(String[] args) {
        String[] files = {"model_1.csv", "model_2.csv", "model_3.csv"};
        float epsilon = 1e-5f;

        String bestMseModel = "", bestMaeModel = "", bestMareModel = "";
        float minMse = Float.MAX_VALUE, minMae = Float.MAX_VALUE, minMare = Float.MAX_VALUE;

        for (String filePath : files) {
            System.out.println("for " + filePath);
            try {
                FileReader filereader = new FileReader(filePath);
                CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
                List<String[]> allData = csvReader.readAll();

                float mse = 0, mae = 0, mare = 0;
                int n = allData.size();

                for (String[] row : allData) {
                    float y_true = Float.parseFloat(row[0]);
                    float y_predicted = Float.parseFloat(row[1]);

                    float error = y_true - y_predicted;
                    mse += (error * error);
                    mae += Math.abs(error);
                    mare += Math.abs(error) / (Math.abs(y_true) + epsilon);
                }

                mse /= n;
                mae /= n;
                mare = (mare / n) * 100;

                System.out.println("\tMSE =" + mse);
                System.out.println("\tMAE =" + mae);
                System.out.println("\tMARE =" + mare);

                if (mse < minMse) { minMse = mse; bestMseModel = filePath; }
                if (mae < minMae) { minMae = mae; bestMaeModel = filePath; }
                if (mare < minMare) { minMare = mare; bestMareModel = filePath; }

            } catch (Exception e) {
                System.out.println("Error reading the CSV file: " + filePath);
            }
        }

        System.out.println("According to MSE, The best model is " + bestMseModel);
        System.out.println("According to MAE, The best model is " + bestMaeModel);
        System.out.println("According to MARE, The best model is " + bestMareModel);
    }
}