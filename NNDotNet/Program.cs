// See https://aka.ms/new-console-template for more information

Console.WriteLine("Hello, World!");
var random = new Random();
var inputSize = 784;
var hiddenSize = 64;
var outputSize = 10;
double[,] W1 = new double[inputSize, hiddenSize];
double[,] W2 = new double[hiddenSize, outputSize];
double[] b1 = new double[hiddenSize];
double[] b2 = new double[outputSize];
//For each percorre em i e itera ate 784, salvando no neurônio J
//Exemplo: pixel 0, w1[0, n0]
for (int i = 0; i < inputSize; i++)
{
    for (int j = 0; j < hiddenSize; j++)
    {
        W1[i, j] = (random.NextDouble() * 2 - 1) * 0.01;
    }
}

double backward(
    double[] x,
    double[] hiddenOutput,
    double[] outputOutput,
    double[] input,
    double learningRate = 0.1)
{
    double[] error = new double[outputSize];
    double[] delta2 = new double[outputSize];
    double[] delta1 = new double[hiddenSize];

    for (int i = 0; i < outputSize; i++)
    {
        error[i] = outputOutput[i] - input[i]; 
    }

    var sigmoidDerivativeOutput = SigmoidDerivate(outputOutput);
    for (int i = 0; i < outputSize; i++)
    {
        delta2[i] = error[i] * sigmoidDerivativeOutput[i];
    }

    for (int i = 0; i < hiddenSize; i++)
    {
        double soma = 0;
        for (int j = 0; j < outputSize; j++)
        {
            soma += delta2[j] * W2[i, j];
        }

        delta1[i] = soma;
    }

    var sigmoidDerivativeHiddenOutput = SigmoidDerivate(hiddenOutput);

    for (int i = 0; i < hiddenSize; i++)
    {
        delta1[i] = error[i] * sigmoidDerivativeHiddenOutput[i];
    }

    for (int i = 0; i < hiddenSize; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            W2[i, j] -= learningRate * hiddenOutput[i] * sigmoidDerivativeOutput[j];
        }
    }

    for (int i = 0; i < outputSize; i++)
    {
        b2[i] -= learningRate * delta2[i];
    }

    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < hiddenSize; j++)
        {
            W1[i, j] -= learningRate * x[i] * delta1[j];
        }
    }
    for (int i = 0; i < hiddenSize; i++)
    {
        b1[i] -= learningRate * delta1[i];
    }

    double loss = 0;
    for (int i = 0; i < outputSize; i++)
    {
        loss += error[i] * error[i];
    }

    loss /= outputSize;
    return loss;
}
(double[] hiddenOutput, double[] outputOutput) Forward(double[] x)
{
    var z1 = ProdutoEscalar(x, W1);
    var a1 = Sigmoid(z1);
    var z2 = ProdutoEscalar(a1, W2);
    var a2 = Sigmoid(z2); 
    return (a1, a2);
}
//Iterar sobre os neurônios normalizando com sigmoid
double[] Sigmoid(double[] z)
{
    double[] resultado = new double[z.Length];
    for (int i = 0; i < z.Length; i++)
    {
        resultado[i] = 1.0 / (1.0 + Math.Exp(-z[i]));
    }
    return resultado;
}

double[] SigmoidDerivate(double[] x)
{
    double[] resultado = new double[x.Length];
    for (int i = 0; i < x.Length; i++)
    {
        resultado[i] = x[i] * (1 - x[i]);
    }
    return resultado;
}
//Itera sobre a hidden layer, pega o bias do neuronio, e soma com o input(Mnist) * peso(W1[i])
double[] ProdutoEscalar(double[] x, double[,] weights)
{
    double[] z1 = new double[hiddenSize];
    for (int j = 0; j < hiddenSize; j++)
    {
        double soma = b1[j];
        for (int i = 0; i < inputSize; i++)
        {
            soma += x[i] * weights[i, j];
        }

        z1[j] = soma;
    }
    return z1; 
}
