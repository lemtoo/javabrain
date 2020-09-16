package ponder.main;

import ponder.activation.SigmoidActivation;
import ponder.activation.TanHActivation;
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
      // InputSet inputSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_Inputs.csv");
      // InputSet testSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_TestSet.csv");

      // InputSet inputSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_Inputs_bigger.csv");
      // InputSet testSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_TestSet_bigger.csv");

      InputSet inputSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_Inputs_64k.csv");
      InputSet testSet = new InputSet("NeuralNet/src/main/resources/xor/XOR_TestSet_bigger.csv");

      final List<String> inputFeatureNames = Arrays.asList("x1", "x2", "Random1", "Random2", "Random3", "Random4", "Random5", "Random6");

      final NeuralNetTester neuralNetTester = new NeuralNetTester();

      /*
      With the failure threshold at 0.1 and 256 rows in both the training set and the test set
      and epoch counts {1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000} run 1000 times each
      The neural net (2 layers with 2 and 1 neurons, respectively) produced the following results:
        For 1000, Failure count is 1000(100.0%)
        For 5000, Failure count is 725(72.5%)
        For 10000, Failure count is 260(26.0%)
        For 20000, Failure count is 184(18.4%)
        For 40000, Failure count is 165(16.5%)
        For 60000, Failure count is 160(16.0%)
        For 80000, Failure count is 171(17.1%)
        For 100000, Failure count is 152(15.2%)
        For 200000, Failure count is 160(16.0%)
        For 500000, Failure count is 141(14.1%)

      With the failure threshold at 0.1 and 4096 rows in both the training set and the test set
      and epoch counts {1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000} run 1000 times each
      The neural net (2 layers with 2 and 1 neurons, respectively) produced the following results:
        For 1000, Failure count is 1000(100.0%)
        For 5000, Failure count is 788(78.8%)
        For 10000, Failure count is 320(32.0%)
        For 20000, Failure count is 201(20.1%)
        For 40000, Failure count is 194(19.4%)
        For 60000, Failure count is 202(20.2%)
        For 80000, Failure count is 185(18.5%)
        For 100000, Failure count is 175(17.5%)
        For 200000, Failure count is 177(17.7%)
        For 500000, Failure count is 180(18.0%)

      With the failure threshold at 0.1, 65536 rows in the training set, and 4096 rows in the test set
      and epoch counts {1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000} run 1000 times each
      The neural net (2 layers with 2 and 1 neurons, respectively) produced the following results:
        For 1000, Failure count is 1000(100.0%)
        For 5000, Failure count is 790(79.0%)
        For 10000, Failure count is 316(31.6%)
        For 20000, Failure count is 202(20.2%)
        For 40000, Failure count is 190(19.0%)
        For 60000, Failure count is 180(18.0%)
        For 80000, Failure count is 173(17.3%)
        For 100000, Failure count is 187(18.7%)
        For 200000, Failure count is 185(18.5%)
        For 500000, Failure count is 187(18.7%)

      After changing the learning rate from 0.3 to 0.1
      With the failure threshold at 0.1, 65536 rows in the training set, and 4096 rows in the test set
      and epoch counts {1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000} run 1000 times each
      The neural net (2 layers with 2 and 1 neurons, respectively) produced the following results:
        For 1000, Failure count is 1000(100.0%)
        For 5000, Failure count is 1000(100.0%)
        For 10000, Failure count is 983(98.3%)
        For 20000, Failure count is 551(55.1%)
        For 40000, Failure count is 251(25.1%)
        For 60000, Failure count is 236(23.6%)
        For 80000, Failure count is 209(20.9%)
        For 100000, Failure count is 189(18.9%)
        For 200000, Failure count is 198(19.8%)
        For 500000, Failure count is 191(19.1%)

      With learning rate at 0.3 and a TanH activation
      With the failure threshold at 0.1, 65536 rows in the training set, and 4096 rows in the test set
      and epoch counts {1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000} run 1000 times each
      The neural net (2 layers with 2 and 1 neurons, respectively) produced the following results:
        For 1000, Failure count is 530(53.0%)
        For 5000, Failure count is 197(19.7%)
        For 10000, Failure count is 155(15.5%)
        For 20000, Failure count is 140(14.0%)
        For 40000, Failure count is 119(11.9%)
        For 60000, Failure count is 125(12.5%)
        For 80000, Failure count is 119(11.9%)
        For 100000, Failure count is 118(11.8%)
        For 200000, Failure count is 113(11.3%)
        For 500000, Failure count is 116(11.6%)
       */

      // Add or remove numbers to test different epoch counts.
      int[] epochCounts = new int[]{1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 200000, 500000};
      // Reduce this number if you need to examine a subset of the epoch counts.
      final int numTests = epochCounts.length;
      // This will count the number of failures per epoch count.
      int[] failureCounts = new int[numTests];
      // This determines the minimum MSE for a run to be considered a failure.
      double failureThreshold = 0.1;
      // This is the number of complete tests per epoch count. In other words, this is the number of times
      // that the neural net will be recreated and run over the given number of epochs.
      int numRuns = 1000;
      // This will accumulate the MSE for each run, grouped by the number of epochs per run.
      double[][] msePerRun = new double[numTests][];

      // For each number of epochs to train...
      for (int runSet = 0; runSet < numTests; runSet++)
      {
        msePerRun[runSet] = new double[numRuns];

        System.out.println("Starting test run for " + epochCounts[runSet] + " epochs.");

        // ...run the required number of test runs by creating a new neural net and running it for the
        // required number of epochs.
        int failCount = 0;
        for (int testRun = 0; testRun < numRuns; testRun++)
        {
          // Create a neural net and a trainer.
          NeuralNetwork neuralNetwork = createNeuralNet(inputFeatureNames.size());
          NetworkTrainer trainer = new NetworkTrainer(neuralNetwork, inputFeatureNames, "Expected");

          // Train the neural net.
          double mse = neuralNetTester.runNeuralNet(neuralNetwork, epochCounts[runSet], trainer, inputSet, testSet);
          // Count failures.
          if (mse > failureThreshold)
          {
            failCount++;
          }
          // Remember the MSE.
          msePerRun[runSet][testRun] = mse;
        }

        // Remember the failure count.
        failureCounts[runSet] = failCount;
      }

      // Print all MSEs.
      for (int runSet = 0; runSet < numTests; runSet++)
      {
        for(int testRun = 0; testRun < numRuns; testRun++)
        {
          System.out.println("Run " + testRun + " MSE: " + msePerRun[runSet][testRun]);
        }
      }

      // Summarize by printing the number of failures for each epoch count.
      for (int runSet = 0; runSet < numTests; runSet++)
      {
        int numEpochs = epochCounts[runSet];
        int failCount = failureCounts[runSet];
        double failPercent = 100.0d * (double)failCount / (double)numRuns;
        System.out.println("For " + numEpochs + ", Failure count is " + failCount + "(" +  failPercent + "%)");
      }

    } catch (PonderInvalidInputException e)
    {
      e.printStackTrace();
    }
  }

  private static NeuralNetwork createNeuralNet(int numFeatures) throws PonderInvalidInputException
  {
    // Create a network...
    NeuralNetwork neuralNetwork = new NeuralNetwork();
    // ...with a layer using a sigmoid activation and 4 neurons, which takes 2 inputs...
    //neuralNetwork.addLayer(4, 3, new SigmoidActivation());
    // ...and a layer which consolidates those inputs into a single output.
    neuralNetwork.addLayer(2, numFeatures, new TanHActivation());
    neuralNetwork.addLayer(1, 2, new TanHActivation
        ());

    return neuralNetwork;
  }
}
