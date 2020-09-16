package ponder.activation;

public class TanHActivation implements Activation
{
  @Override
  public double activate(double input)
  {
    return (2.0 / (1.0 + Math.exp(-2.0 * input))) - 1.0;
  }

  @Override
  public double calculateDerivative(double input)
  {
    return 1.0 - (input * input);
  }
}
