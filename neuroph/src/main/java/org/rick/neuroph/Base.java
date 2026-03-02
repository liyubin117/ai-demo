package org.rick.neuroph;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;

import java.util.Arrays;

public class Base {
    public static void main(String[] args) {
        // 创建神经网络
        MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(2, 3, 1);

// 创建训练数据
        DataSet trainingSet = new DataSet(2, 1);
        DataSetRow row1 = new DataSetRow(new double[]{0, 0}, new double[]{0});
        trainingSet.add(row1);
        trainingSet.add(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.add(new DataSetRow(new double[]{1, 1}, new double[]{0}));

// 训练网络
        neuralNetwork.learn(trainingSet);

// 使用网络
        neuralNetwork.setInput(1, 1);
        neuralNetwork.calculate();
        double[] networkOutput = neuralNetwork.getOutput();

        System.out.println(Arrays.toString(networkOutput));
    }
}
