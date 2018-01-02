using Encog.MathUtil.RBF;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Data.Basic;
using Encog.Neural.Pattern;
using Encog.Neural.Rbf.Training;
using Encog.Neural.RBF;
using System;

namespace rbf
{
    class RbfTester
    {
        double[][] input;
        double[][] ideal;

        double[][] testInput;
        double[][] testIdeal;

        IMLData tInput;
        IMLData tIdeal;

        double xMaxScale;
        double xMinScale;
        double yMaxScale;
        double yMinScale;

        private BasicNeuralDataSet trainingSet;

        public RbfTester()
        {
        }

        public void run(int neurons, int trainInputCount)
        {
            int dimensions = 1;
            int numNeuronsPerDimension = neurons;
            double volumeNeuronWidth = 2.0 / numNeuronsPerDimension;
            bool includeEdgeRBFs = true;

            var pattern = new RadialBasisPattern();
            pattern.InputNeurons = dimensions;
            pattern.OutputNeurons = 1;
            int numNeurons = (int)Math.Pow(numNeuronsPerDimension, dimensions);
            pattern.AddHiddenLayer(numNeurons);

            var network = (RBFNetwork)pattern.Generate();
            Console.Out.WriteLine("num of hideen neurons: " + numNeurons + ", train input len: " + trainInputCount);

            network.SetRBFCentersAndWidthsEqualSpacing(0, 1, RBFEnum.Gaussian, volumeNeuronWidth, includeEdgeRBFs);

            initInputs(trainInputCount, trainInputCount / 3);

            IMLDataSet trainingSet = new BasicNeuralDataSet(input, ideal);
            SVDTraining train = new SVDTraining(network, trainingSet);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            int epoch = 1;
            do
            {
                train = new SVDTraining(network, trainingSet);
                train.Iteration();
                Console.WriteLine("Epoch #" + epoch + " Error:" + train.Error);
                epoch++;
            } while ((epoch < 1) && (train.Error > 0.001));
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.Out.WriteLine("Training time: " + elapsedMs + " ms");

            IMLDataSet testingSet = new BasicMLDataSet(testInput, testIdeal);


            double errorSum = 0.0;
            int count = 0;
            foreach (BasicMLDataPair pair in testingSet)
            {
                BasicMLData output = (BasicMLData)network.Compute(pair.Input);
                double error = Math.Pow(output.Data[0] - pair.Ideal[0], 2);
                errorSum += error;

                //Console.Out.WriteLine("ideal: " + pair.Ideal[0] + ", out: " + output.Data[0]);

                count++;
            }
            
            Console.Out.WriteLine("test error: " + errorSum / count);
            Console.Out.WriteLine("--------------------------------------------------------------");
        }

        private double f(double x)
        {
            return Math.Sin(x * Math.Sin(x));
        }

        private void initInputs(int trainLen, int testLen)
        {
            input = new double[trainLen][];
            ideal = new double[trainLen][];

            testInput = new double[testLen][];
            testIdeal = new double[testLen][];

            double x = 0.0;
            double temp;
            xMaxScale = xMinScale = x;
            yMaxScale = xMinScale = f(x);
            for (int i = 0; i < trainLen; i++)
            {
                input[i] = new double[1];
                ideal[i] = new double[1];
                temp = f(x);

                input[i][0] = x;
                ideal[i][0] = temp;

                if (x > xMaxScale) xMaxScale = x;
                if (x < xMinScale) xMinScale = x;
                if (temp > yMaxScale) yMaxScale = temp;
                if (temp < yMinScale) yMinScale = temp;

                x += 0.1;
            }

            for (int i = 0; i < trainLen; i++)
            {
                input[i][0] = (input[i][0] - xMinScale) / (xMaxScale - xMinScale);
                ideal[i][0] = (ideal[i][0] - yMinScale) / (yMaxScale - yMinScale);
            }

            x = 0.0;
            for(int i = 0; i < testLen; i++)
            {
                testInput[i] = new double[1];
                testIdeal[i] = new double[1];

                testInput[i][0] = (x - xMinScale) / (xMaxScale - xMinScale);
                testIdeal[i][0] = (f(x) - yMinScale) / (yMaxScale - yMinScale);

                x += 0.17;
            }
        }
    }
}
