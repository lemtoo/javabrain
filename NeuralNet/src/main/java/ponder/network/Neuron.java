package ponder.network;

import ponder.activation.Activation;
import ponder.err.PonderInvalidInputException;

import java.util.Arrays;
import java.util.Random;

/**
 * A Neuron is a single unit of calculation within the neural net. Each neuron semi-randomly develops an effect as
 * the neural net back-propagates error.
 */
public class Neuron
{
  private double bias;
  private double[] weights;

  private double[] lastInputSet;
  private double lastOutput;
  private double lastError = 0d;

  /**
   * Initialize this Neuron with the specified parameters.
   *
   * @param bias The initial bias for this neuron.
   * @param weights The initial weight list for this neuron. This array will be copied, not used as-is.
   * @throws PonderInvalidInputException Indicates invalid neuron configuration.
   */
  public Neuron(double bias, double[] weights) throws PonderInvalidInputException
  {
    if(weights == null)
    {
      throw new PonderInvalidInputException("Neuron initial weight matrix is null!");
    }

    if(weights.length == 0)
    {
      throw new PonderInvalidInputException("Neuron initial weight matrix is empty!");
    }

    this.bias = bias;
    this.weights = Arrays.copyOf(weights, weights.length);
  }

  /**
   * Initialize this Neuron with the specified number of randomly generated weights in the range [-1,1] and a
   * randomly generated bias in the same range.
   *=
   * @param numWeights The number of weights to randomly initialize.
   * @param activatorFunction The activator function for this neuron.
   * @throws PonderInvalidInputException Indicates invalid neuron configuration.
   */
  public Neuron(int numWeights, Activation activatorFunction) throws PonderInvalidInputException
  {
    if(numWeights == 0)
    {
      throw new PonderInvalidInputException("Neuron initial weight count is empty!");
    }

    final Random rng = new Random();

    bias = 1.0d;

    // In this case, initialize the weight matrix with small random values.
    weights = new double[numWeights];
    for (int weightIndex = 0, weightsLength = weights.length; weightIndex < weightsLength; weightIndex++)
    {
      // Random number in [-1, 1]
      weights[weightIndex] = (rng.nextDouble() * 2.0d) - 1.0d;
    }
  }

  /**
   * Fire this neuron. This method takes the inputs, applies the weights and bias, and then runs the activator
   * function.
   *
   * @param inputs The inputs, which must be equal in number to the weights.
   * @return The calculated output value.
   * @throws PonderInvalidInputException Indicates that an invalid set of inputs was provided.
   */
  public double fire(double[] inputs, Activation activatorFunction) throws PonderInvalidInputException
  {
    if(inputs == null)
    {
      throw new PonderInvalidInputException("Neuron input is null!");
    }

    if(inputs.length != weights.length)
    {
      throw new PonderInvalidInputException("Neuron input count does not match weight count!");
    }

    lastInputSet = inputs;

    double result = bias;

    for(int i=0; i<weights.length; i++)
    {
      result += weights[i] * inputs[i];
    }

    lastOutput = activatorFunction.activate(result);
    return lastOutput;
  }

  /**
   *
   * @return The number of expected inputs.
   */
  public int getWeightCount()
  {
    return weights.length;
  }

  public double getLastOutput()
  {
    return lastOutput;
  }

  public void setLastError(double lastError)
  {
    this.lastError = lastError;
  }

  public double getLastError()
  {
    return lastError;
  }

  public double getWeight(int index)
  {
    return weights[index];
  }

  public double getBias()
  {
    return bias;
  }

  public void applyError(double learningRate)
  {
    final double updateAmount = learningRate * lastError;

    for (int i = 0; i < weights.length; i++)
    {
      weights[i] += updateAmount * lastInputSet[i];
    }
    bias += updateAmount;
  }
}
