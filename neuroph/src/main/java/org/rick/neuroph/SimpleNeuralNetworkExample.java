package org.rick.neuroph;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.PerceptronLearning;

import java.util.Arrays;

/**
 * 简单神经网络示例：感知机实现逻辑AND运算
 * 作者：基于《自己动手写神经网络》书中示例
 */
public class SimpleNeuralNetworkExample {

    public static void main(String[] args) {

        System.out.println("=== 简单神经网络示例：感知机实现逻辑AND运算 ===\n");

        // 1. 创建神经网络：2个输入神经元，1个输出神经元
        System.out.println("1. 创建感知机神经网络（2输入，1输出）...");
        NeuralNetwork<PerceptronLearning> neuralNetwork = new Perceptron(2, 1);

        // 2. 创建训练数据集
        System.out.println("2. 创建训练数据集...");
        DataSet trainingSet = new DataSet(2, 1);

        // 3. 添加训练样本：逻辑AND的真值表
        // 输入：[0,0] -> 输出：[0]
        // 输入：[0,1] -> 输出：[0]
        // 输入：[1,0] -> 输出：[0]
        // 输入：[1,1] -> 输出：[1]
        trainingSet.add(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.add(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet.add(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet.add(new DataSetRow(new double[]{1, 1}, new double[]{1}));

        System.out.println("训练数据集大小: " + trainingSet.size() + " 个样本");

        // 4. 获取学习规则并设置参数
        System.out.println("3. 配置学习参数...");
        PerceptronLearning learningRule = neuralNetwork.getLearningRule();
        learningRule.setMaxError(0.01);     // 最大允许误差
        learningRule.setMaxIterations(1000); // 最大迭代次数
        learningRule.setLearningRate(0.1);   // 学习率

        System.out.println("   - 最大允许误差: " + learningRule.getMaxError());
        System.out.println("   - 最大迭代次数: " + learningRule.getMaxIterations());
        System.out.println("   - 学习率: " + learningRule.getLearningRate());

        // 5. 训练神经网络
        System.out.println("\n4. 开始训练神经网络...");
        neuralNetwork.learn(trainingSet);

        System.out.println("训练完成！");
        System.out.println("实际迭代次数: " + learningRule.getCurrentIteration());
        System.out.println("最终误差: " + learningRule.getTotalNetworkError() + "\n");

        // 6. 测试训练好的神经网络
        System.out.println("5. 测试神经网络性能：");
        System.out.println("=======================");

        testNeuralNetwork(neuralNetwork, new double[]{0, 0}, "0 AND 0");
        testNeuralNetwork(neuralNetwork, new double[]{0, 1}, "0 AND 1");
        testNeuralNetwork(neuralNetwork, new double[]{1, 0}, "1 AND 0");
        testNeuralNetwork(neuralNetwork, new double[]{1, 1}, "1 AND 1");

        // 7. 测试一些边界情况
        System.out.println("\n6. 测试边界情况：");
        System.out.println("====================");
        testNeuralNetwork(neuralNetwork, new double[]{0.5, 0.5}, "0.5 AND 0.5");
        testNeuralNetwork(neuralNetwork, new double[]{0.8, 0.2}, "0.8 AND 0.2");
        testNeuralNetwork(neuralNetwork, new double[]{0.9, 0.9}, "0.9 AND 0.9");

        // 8. 显示网络权重
        System.out.println("\n7. 神经网络最终权重：");
        System.out.println("=======================");
        Double[] weights = neuralNetwork.getWeights();
        System.out.println("权重 w1: " + weights[0]);
        System.out.println("权重 w2: " + weights[1]);
        System.out.println(Arrays.toString(weights));
    }

    /**
     * 测试神经网络对给定输入的输出
     */
    private static void testNeuralNetwork(NeuralNetwork neuralNetwork, double[] input, String testName) {
        neuralNetwork.setInput(input);
        neuralNetwork.calculate();
        double[] output = neuralNetwork.getOutput();

        // 由于感知机输出是0或1，我们进行四舍五入
        double predictedValue = output[0];
        double roundedValue = Math.round(predictedValue);

        System.out.printf("%-15s -> 输入: [%.1f, %.1f], 原始输出: %.4f, 四舍五入: %.0f (预测: %s)%n",
                testName, input[0], input[1], predictedValue, roundedValue,
                roundedValue > 0.5 ? "TRUE" : "FALSE");
    }
}
