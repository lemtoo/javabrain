package ponder.main;

import ponder.activation.SigmoidActivation;
import ponder.err.PonderInvalidInputException;
import ponder.network.NetworkTrainer;
import ponder.network.NeuralNetwork;

import java.util.Arrays;

public class NetworkTester
{
  public static void main(String[] args)
  {
    try
    {
      // Create a network...
      NeuralNetwork neuralNetwork = new NeuralNetwork();
      // ...with a layer using a sigmoid activation and 4 neurons, which takes 2 inputs...
      //neuralNetwork.addLayer(4, 3, new SigmoidActivation());
      // ...and a layer which consolidates those inputs into a single output.
      neuralNetwork.addLayer(5, 5, new SigmoidActivation());
      neuralNetwork.addLayer(1, 5, new SigmoidActivation());

      // Create a trainer.
      NetworkTrainer trainer = new NetworkTrainer(neuralNetwork, Arrays.asList(
          "x1", "x2", "unimportant1", "unimportant2", "unimportant3"), "xor");

      // Set up data to train on the "XOR" function.
      double[][] xorSamples = {
          {0d, 0d, .5d,  1d, .6d, 0d},
          {0d, 1d,  1d, .2d, .2d, 1d},
          {1d, 0d, .2d, .7d,  1d, 1d},
          {1d, 1d, .9d, .5d,  0d, 0d},

          {0d, 0d, 0d, .1d, .3d, 0d},
          {0d, 1d, 0d, .6d, .2d, 1d},
          {1d, 0d, 0d, .7d, .1d, 1d},
          {1d, 1d, 0d, .8d, .9d, 0d},

          {0d, 0d, 1d, 1d, .2d, 0d},
          {0d, 1d, 1d, 1d, .2d, 1d},
          {1d, 0d, 1d, 1d, .1d, 1d},
          {1d, 1d, 1d, 1d, .5d, 0d},

          {0d, 0d, .5d, .7d, .6d, 0d},
          {0d, 1d, .1d, .2d, .2d, 1d},
          {1d, 0d, .2d, .7d, .1d, 1d},
          {1d, 1d, .9d, .5d, .5d, 0d}
      };
      trainer.setTrainingData(new String[]{"x1", "x2", "unimportant1", "unimportant2", "unimportant3",  "xor"}, xorSamples);

      // Provide the same sample data without the expected result.
      double[][] xorTester = {
          {0d, 0d, .5d, 1d, 0d},
          {0d, 1d, .2d, 1d, 0d},
          {1d, 0d, .9d, 1d, 0d},
          {1d, 1d, .3d, 1d, 0d}
      };

      System.out.println("Initial Weights:");
      System.out.println(neuralNetwork.printWeights());

      System.out.println("Initial run results:");
      for (double[] xorSample : xorTester)
      {
        final double[] runResult = neuralNetwork.run(xorSample);
        for (int i = 0, runResultLength = runResult.length; i < runResultLength; i++)
        {
          double result = runResult[i];
          System.out.println(i + ": " + result);
        }
      }

      System.out.println("Weights per Run:");
      for(int i=0 ;i<10000; i++)
      {
        // And make one training attempt.
        trainer.runOnce();

        System.out.println(neuralNetwork.printWeights());
      }

      System.out.println("Final Weights:");
      System.out.println(neuralNetwork.printWeights());

      System.out.println("Final run results:");
      for (double[] xorSample : xorTester)
      {
        final double[] runResult = neuralNetwork.run(xorSample);
        for (int i = 0, runResultLength = runResult.length; i < runResultLength; i++)
        {
          double result = runResult[i];
          System.out.println(i + ": " + result);
        }
      }
    }
    catch (PonderInvalidInputException e)
    {
      e.printStackTrace();
    }
  }
}
