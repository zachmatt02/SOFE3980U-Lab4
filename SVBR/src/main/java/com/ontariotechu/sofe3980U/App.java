package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;

public class App {
    public static void main(String[] args) {
        String[] files = {"model_1.csv", "model_2.csv", "model_3.csv"};

        String bestBceModel="", bestAccModel="", bestPrecModel="", bestRecModel="", bestF1Model="", bestAucModel="";
        float minBce = Float.MAX_VALUE, maxAcc = 0, maxPrec = 0, maxRec = 0, maxF1 = 0, maxAuc = 0;

        for (String filePath : files) {
            System.out.println("for " + filePath);
            try {
                FileReader filereader = new FileReader(filePath);
                CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
                List<String[]> allData = csvReader.readAll();

                int n = allData.size();
                float bce = 0;
                int TP = 0, FP = 0, TN = 0, FN = 0;
                int n_positive = 0, n_negative = 0;

                for (String[] row : allData) {
                    int y_true = Integer.parseInt(row[0]);
                    float y_predicted = Float.parseFloat(row[1]);

                    if (y_true == 1) {
                        bce += Math.log(y_predicted);
                        n_positive++;
                    } else {
                        bce += Math.log(1 - y_predicted);
                        n_negative++;
                    }

                    int y_hat_binary = (y_predicted >= 0.5) ? 1 : 0;
                    if (y_true == 1 && y_hat_binary == 1) TP++;
                    else if (y_true == 0 && y_hat_binary == 1) FP++;
                    else if (y_true == 0 && y_hat_binary == 0) TN++;
                    else if (y_true == 1 && y_hat_binary == 0) FN++;
                }

                bce = -bce / n;
                float accuracy = (float)(TP + TN) / (TP + TN + FP + FN);
                float precision = (float)TP / (TP + FP);
                float recall = (float)TP / (TP + FN);
                float f1 = 2 * (precision * recall) / (precision + recall);

                float[] x = new float[101];
                float[] y = new float[101];

                for (int i = 0; i <= 100; i++) {
                    float th = i / 100.0f;
                    int tp_th = 0, fp_th = 0;
                    for (String[] row : allData) {
                        int y_true = Integer.parseInt(row[0]);
                        float y_predicted = Float.parseFloat(row[1]);
                        if (y_true == 1 && y_predicted >= th) tp_th++;
                        if (y_true == 0 && y_predicted >= th) fp_th++;
                    }
                    y[i] = (float)tp_th / n_positive;
                    x[i] = (float)fp_th / n_negative;
                }

                float auc = 0;
                for (int i = 1; i <= 100; i++) {
                    auc += (y[i - 1] + y[i]) * Math.abs(x[i - 1] - x[i]) / 2.0;
                }

                System.out.println("\tBCE =" + bce);
                System.out.println("\tConfusion matrix");
                System.out.println("\t\t\ty=1\ty=0");
                System.out.println("\t\ty^=1\t" + TP + "\t" + FP);
                System.out.println("\t\ty^=0\t" + FN + "\t" + TN);
                System.out.println("\tAccuracy =" + accuracy);
                System.out.println("\tPrecision =" + precision);
                System.out.println("\tRecall =" + recall);
                System.out.println("\tf1 score =" + f1);
                System.out.println("\tauc roc =" + auc);

                if (bce < minBce) { minBce = bce; bestBceModel = filePath; }
                if (accuracy > maxAcc) { maxAcc = accuracy; bestAccModel = filePath; }
                if (precision > maxPrec) { maxPrec = precision; bestPrecModel = filePath; }
                if (recall > maxRec) { maxRec = recall; bestRecModel = filePath; }
                if (f1 > maxF1) { maxF1 = f1; bestF1Model = filePath; }
                if (auc > maxAuc) { maxAuc = auc; bestAucModel = filePath; }

            } catch (Exception e) {
                System.out.println("Error reading the CSV file: " + filePath);
            }
        }

        System.out.println("According to BCE, The best model is " + bestBceModel);
        System.out.println("According to Accuracy, The best model is " + bestAccModel);
        System.out.println("According to Precision, The best model is " + bestPrecModel);
        System.out.println("According to Recall, The best model is " + bestRecModel);
        System.out.println("According to F1 score, The best model is " + bestF1Model);
        System.out.println("According to AUC ROC, The best model is " + bestAucModel);
    }
}