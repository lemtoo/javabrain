package ponder.main;

import ponder.activation.SigmoidActivation;
import ponder.data.InputSet;
import ponder.err.PonderInvalidInputException;
import ponder.network.NetworkTrainer;
import ponder.network.NeuralNetwork;

import java.util.Arrays;
import java.util.List;

/**
 * This example class constructs a 2-layer neural network with 8 features and 1 target. Of the 8 features, two have a
 * direct XOR relationship to the target. The first layer has 2 neurons, and the second layer has 1 neuron; any fewer
 * layers or neurons is ineffective, but more neurons does not seem to have a consistent positive effect.
 *
 * This neural network uses a sigmoid activation function. Despite there being a direct relationship between the
 * convolution of two features and the target, a linear activation does not produce good results.
 */
public class XORExample
{
  public static void main(String[] args)
  {
    try
    {
      // Load up some data.
      InputSet inputSet = new InputSet("NeuralNet/src/main/resources/XOR_Inputs.csv");
      InputSet testSet = new InputSet("NeuralNet/src/main/resources/XOR_TestSet.csv");

      final List<String> inputFeatureNames = Arrays.asList("x1", "x2", "Random1", "Random2", "Random3", "Random4", "Random5", "Random6");

      // Create a network...
      NeuralNetwork neuralNetwork = new NeuralNetwork();
      // ...with a layer using a sigmoid activation and 4 neurons, which takes 2 inputs...
      //neuralNetwork.addLayer(4, 3, new SigmoidActivation());
      // ...and a layer which consolidates those inputs into a single output.
      neuralNetwork.addLayer(2, inputFeatureNames.size(), new SigmoidActivation());
      neuralNetwork.addLayer(1, 2, new SigmoidActivation());

      // Create a trainer.
      NetworkTrainer trainer = new NetworkTrainer(neuralNetwork, inputFeatureNames, "Expected");

      trainer.setTrainingData(inputSet.getFeatureNames(), inputSet.getInputs());
      System.out.println("Initial Weights:");
      System.out.println(neuralNetwork.printWeights());

      runTests(neuralNetwork, testSet);

      //System.out.println("Weights per Run:");
      for (int i = 0; i < 20000; i++)
      {
        // And make one training attempt.
        trainer.runOnce();

        //System.out.println(neuralNetwork.printWeights());
      }

      System.out.println("Final Weights:");
      System.out.println(neuralNetwork.printWeights());

      runTests(neuralNetwork, testSet);

    } catch (PonderInvalidInputException e)
    {
      e.printStackTrace();
    }
  }

  private static void runTests(NeuralNetwork neuralNetwork, InputSet testSet) throws PonderInvalidInputException
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
      if(i < 4)
      {
        System.out.println("Run result: " + runResult[0] + "; Error: " + error);
      }
    }
    mse /= (double)filteredInputFeatures.length;
    meanErr /= (double)filteredInputFeatures.length;
    System.out.println("Test run results: MSE is [" + mse + "], and Mean Error is [" + meanErr + "]");
  }
}
