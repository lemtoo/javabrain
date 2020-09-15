package ponder.network;

import ponder.activation.Activation;
import ponder.activation.SigmoidActivation;
import ponder.err.PonderInvalidInputException;

import java.util.ArrayList;

public class Layer
{
  private ArrayList<Neuron> neurons = new ArrayList<>();
  private final Activation activatorFunction;

  public Layer()
  {
    activatorFunction = new SigmoidActivation();
  }

  public Layer(int numNeurons, int numInputs, Activation activatorFunction, Activation activatorFunction1) throws PonderInvalidInputException
  {
    this.activatorFunction = activatorFunction1;
    addNeurons(numInputs, activatorFunction, numNeurons);
  }

  public void addNeuron(Neuron newNeuron)
  {
    neurons.add(newNeuron);
  }

  /**
   * Generate the specified number of Neurons with the given number of inputs and the given activation function. The
   * weights and bias for the neuron will be randomly generated.
   *
   * @param numInputs The number of inputs expected for this layer.
   * @param activatorFunction The activator function for the new neuron.
   * @param numNeurons The number of Neurons to add with the current parameters.
   * @throws PonderInvalidInputException Indicates an invalid input during Neuron creation.
   */
  public void addNeurons(int numInputs, Activation activatorFunction, int numNeurons) throws PonderInvalidInputException
  {
    for(int counter = 0; counter < numNeurons; counter++)
    {
      addNeuron(numInputs, activatorFunction);
    }
  }

  /**
   * Generate a Neuron with the given number of inputs and the given activation function. The weights and bias for
   * the neuron will be randomly generated.
   *
   * @param numInputs The number of inputs expected for this layer.
   * @param activatorFunction The activator function for the new neuron.
   * @throws PonderInvalidInputException Indicates an invalid input during Neuron creation.
   */
  public void addNeuron(int numInputs, Activation activatorFunction) throws PonderInvalidInputException
  {
    addNeuron(new Neuron(numInputs, activatorFunction));
  }

  public ArrayList<Neuron> getNeurons()
  {
    return neurons;
  }

  /**
   * Fire each of the neurons in this layer. Each neuron should expect the same number of inputs (which should match
   * the number of outputs from the previous layer or, if this is the first layer, the number of features).
   *
   * @param inputs The input values with which to calculate outputs.
   * @return The array of output values.
   * @throws PonderInvalidInputException Indicates invalid input (a null or empty list).
   */
  public double[] fireLayer(double[] inputs) throws PonderInvalidInputException
  {
    if(inputs == null)
    {
      throw new PonderInvalidInputException("Layer input list is null!");
    }
    if(inputs.length == 0)
    {
      throw new PonderInvalidInputException("Layer input list is empty!");
    }

    double[] results = new double[neurons.size()];

    for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++)
    {
      //System.out.println("Firing neuron.");
      results[neuronIndex] = neurons.get(neuronIndex).fire(inputs, activatorFunction);
      //System.out.println("Neuron fired.");
    }

    return results;
  }

  /**
   *
   * @return The number of neurons on this layer, which is also the number of outputs.
   */
  public int getNeuronCount()
  {
    return neurons.size();
  }

  /**
   *
   * @return The expected number of inputs for neurons on this layer.
   */
  public int getInputCount()
  {
    return neurons.get(0).getWeightCount();
  }

  public Activation getActivatorFunction()
  {
    return activatorFunction;
  }
}
