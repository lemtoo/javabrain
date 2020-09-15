package ponder.activation;

public class SigmoidActivation implements Activation
{
  @Override
  public double activate(double input)
  {
    return 1.0 / (1.0 + Math.exp(-input));
  }

  @Override
  public double calculateDerivative(double input)
  {
    return input * (1-input);
  }
}
