package ponder.main;

import ponder.data.InputSet;
import ponder.err.PonderInvalidInputException;
import ponder.network.NetworkTrainer;
import ponder.network.NeuralNetwork;

import java.util.Arrays;
import java.util.List;

public class NeuralNetTester
{
  public double runNeuralNet(NeuralNetwork neuralNetwork, int epochs, NetworkTrainer trainer, InputSet inputSet,
                           InputSet testSet) throws PonderInvalidInputException
  {
    trainer.setTrainingData(inputSet.getFeatureNames(), inputSet.getInputs());

    // This is just for debugging. Uncomment as needed.
    // System.out.println("Initial Weights:");
    // System.out.println(neuralNetwork.printWeights());

    runTests(neuralNetwork, testSet);

    //System.out.println("Weights per Run:");
    for (int i = 0; i < epochs; i++)
    {
      // And make one training attempt.
      trainer.runOnce();

      //System.out.println(neuralNetwork.printWeights());
    }

    // This is just for debugging. Uncomment as needed.
    // System.out.println("Final Weights:");
    // System.out.println(neuralNetwork.printWeights());

    return runTests(neuralNetwork, testSet);
  }

  private static double runTests(NeuralNetwork neuralNetwork, InputSet testSet) throws PonderInvalidInputException
  {
    // Filter out the expected result.
    final List<String> inputFeatureNames = Arrays.asList("x1", "x2", "Random1", "Random2", "Random3", "Random4", "Random5", "Random6");
    final double[][] filteredInputFeatures = testSet.getInputFeatures((String[]) inputFeatureNames.toArray());

    // Run the test set and calculate the MSE.
    double mse = 0d; // Mean Squared Error.
    double meanErr = 0d; // Mean Error
    for (int i = 0; i < filteredInputFeatures.length; i++)
    {
      double[] oneInput = filteredInputFeatures[i];
      final double[] runResult = neuralNetwork.run(oneInput);
      double expectedValue = testSet.getInputs()[i][0];
      double error = expectedValue - runResult[0];
      meanErr += error;
      mse += error * error;
      // This is just for debugging. Uncomment as needed.
      /*
      if(i < 4)
      {
        System.out.println("Run result: " + runResult[0] + "; Error: " + error);
      }
      */
    }
    mse /= (double)filteredInputFeatures.length;
    meanErr /= (double)filteredInputFeatures.length;

    // This is just for debugging. Uncomment as needed.
    // System.out.println("Test run results: MSE is [" + mse + "], and Mean Error is [" + meanErr + "]");

    return mse;
  }
}
