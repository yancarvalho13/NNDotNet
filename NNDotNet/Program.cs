// See https://aka.ms/new-console-template for more information

Console.WriteLine("Hello, World!");

var random = new Random();
var inputSize = 784;
var hiddenSize = 256;
var outputSize = 10;
double[,] W1 = new double[inputSize, hiddenSize];
double[,] W2 = new double[hiddenSize, outputSize];
double[] b1 = new double[hiddenSize];
double[] b2 = new double[outputSize];
var trainingData = LoadMnist(
    "assets/archive/train-images.idx3-ubyte",
    "assets/archive/train-labels.idx1-ubyte")
    .OrderBy(_ => Random.Shared.Next())
    .Take(1000)
    .ToList();

//For each percorre em i e itera ate 784, salvando no neurônio J
//Exemplo: pixel 0, w1[0, n0]
for (int i = 0; i < inputSize; i++)
{
    for (int j = 0; j < hiddenSize; j++)
    {
        W1[i, j] = (random.NextDouble() * 2 - 1) * 0.01;
    }
}
for (int i = 0; i < hiddenSize; i++)
{
    for (int j = 0; j < outputSize; j++)
    {
        W2[i, j] = (random.NextDouble() * 2 - 1) * 0.01;
    }
}

int epochs = 10;
for (int epoch = 0; epoch < epochs; epoch++)
{
    double totalLoss = 0;
    int correct = 0;

    foreach (var sample in trainingData)
    {
        var (hiddenOutput, outputOutput) =
            Forward(sample.Pixels);

        // Predição ANTES de alterar os pesos
        int prediction = Predict(outputOutput);

        if (prediction == sample.Label)
        {
            correct++;
        }

        double loss = Backward(
            sample.Pixels,
            hiddenOutput,
            outputOutput,
            sample.Target,
            learningRate: 0.1
        );

        totalLoss += loss;
    }

    double averageLoss =
        totalLoss / trainingData.Count;

    double accuracy =
        (double)correct / trainingData.Count * 100.0;

    Console.WriteLine(
        $"Epoch {epoch + 1}/{epochs} " +
        $"- Loss: {averageLoss:F4} " +
        $"- Accuracy: {accuracy:F2}%"
    );
}

int Predict(double[] output)
{
    int predicted = 0;

    for (int i = 1; i < output.Length; i++)
    {
        if (output[i] > output[predicted])
        {
            predicted = i;
        }
    }

    return predicted;
}
double Backward(
    double[] x,
    double[] hiddenOutput,
    double[] outputOutput,
    double[] target,
    double learningRate = 0.1)
{
    double[] error = new double[outputSize];

    double[] delta2 = new double[outputSize];
    double[] delta1 = new double[hiddenSize];

    //Erro da saída
    for (int i = 0; i < outputSize; i++)
    {
        error[i] = outputOutput[i] - target[i];
    }

    //Delta da saída

    var sigmoidDerivativeOutput =
        SigmoidDerivate(outputOutput);

    for (int i = 0; i < outputSize; i++)
    {
        delta2[i] =
            error[i] *
            sigmoidDerivativeOutput[i];
    }

    //Propaga o erro para trás
    for (int i = 0; i < hiddenSize; i++)
    {
        double soma = 0;

        for (int j = 0; j < outputSize; j++)
        {
            soma += delta2[j] * W2[i, j];
        }

        delta1[i] = soma;
    }

    //Derivada da hidden

    var sigmoidDerivativeHidden =
        SigmoidDerivate(hiddenOutput);

    for (int i = 0; i < hiddenSize; i++)
    {
        delta1[i] *= sigmoidDerivativeHidden[i];
    }

    //Atualiza W2

    for (int i = 0; i < hiddenSize; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            W2[i, j] -=
                learningRate *
                hiddenOutput[i] *
                delta2[j];
        }
    }

    //Atualiza b2

    for (int i = 0; i < outputSize; i++)
    {
        b2[i] -= learningRate * delta2[i];
    }

    //Atualiza W1

    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < hiddenSize; j++)
        {
            W1[i, j] -=
                learningRate *
                x[i] *
                delta1[j];
        }
    }

    //Atualiza b1

    for (int i = 0; i < hiddenSize; i++)
    {
        b1[i] -= learningRate * delta1[i];
    }

    //MSE

    double loss = 0;

    for (int i = 0; i < outputSize; i++)
    {
        loss += error[i] * error[i];
    }

    return loss / outputSize;
}
(double[] hiddenOutput, double[] outputOutput) Forward(double[] x)
{
    var z1 = ProdutoEscalar(x, W1, b1);
    var a1 = Sigmoid(z1);
    var z2 = ProdutoEscalar(a1, W2, b2);
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
double[] ProdutoEscalar(double[] x, double[,] weights, double[] bias)
{
    int entradas = weights.GetLength(0);
    int saidas  = weights.GetLength(1);
    double[] resultado = new double[saidas];
    for (int j = 0; j < saidas ; j++)
    {
        double soma = bias[j];
        for (int i = 0; i < entradas ; i++)
        {
            soma += x[i] * weights[i, j];
        }

        resultado[j] = soma;
    }
    return resultado; 
}

int ReadBigEndianInt32(BinaryReader reader)
{
    byte[] bytes = reader.ReadBytes(4);
    if(BitConverter.IsLittleEndian)
        Array.Reverse(bytes);
    return BitConverter.ToInt32(bytes, 0);
}

double[][] LoadImages(string path)
{
    using var stream = File.OpenRead(path);
    using var reader = new BinaryReader(stream);
    
    int magic = ReadBigEndianInt32(reader);
    int imageCount = ReadBigEndianInt32(reader);
    int rows = ReadBigEndianInt32(reader);
    int cols = ReadBigEndianInt32(reader);
    
    Console.WriteLine($"Magic: {magic}");
    Console.WriteLine($"ImageCount: {imageCount}");
    Console.WriteLine($"Rows: {rows}");
    Console.WriteLine($"Cols: {cols}");
    
    int pixelsPerImage = rows * cols;
    double[][] images = new double[imageCount][];

    for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
    {
        images[imageIndex] = new double[pixelsPerImage];
        for (int pixelIndex = 0; pixelIndex < pixelsPerImage; pixelIndex++)
        {
            byte pixel = reader.ReadByte();
            images[imageIndex][pixelIndex] = pixel / 255.0;
        }
    }
    return images;
}

byte[] LoadLabels(string path)
{
    using var stream = File.OpenRead(path);
    using var reader = new BinaryReader(stream);

    int magic = ReadBigEndianInt32(reader);
    int labelCount = ReadBigEndianInt32(reader);

    Console.WriteLine($"Magic: {magic}");
    Console.WriteLine($"LabelCount: {labelCount}");

    byte[] labels = new byte[labelCount];
    for (int i = 0; i < labelCount; i++)
    {
        labels[i] = reader.ReadByte();
    }
    return labels;
}

double[] OneHot(byte label)
{
    double[] target = new double[10];
    target[label] = 1.0;
    return target;
}

List<MnistSample> LoadMnist(string imagesPath, string labelPath)
{
    var images = LoadImages(imagesPath);
    var labels = LoadLabels(labelPath);
    
    if(images.Length != labels.Length)
        throw new Exception("Images and labels do not match");
    
    var samples = new List<MnistSample>(images.Length);

    for (int i = 0; i < images.Length; i++)
    {
        samples.Add(new MnistSample
        {
            Pixels = images[i],
            Label = labels[i],
            Target =  OneHot(labels[i])
        });
    }
    
    return samples;
}
public class MnistSample
{
    public double[] Pixels { get; set; } = [];
    public double[] Target { get; set; } = [];
    public byte Label { get; set; }
}
