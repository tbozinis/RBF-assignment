using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.MathUtil.RBF;
using Encog.Neural.NeuralData;
using Encog.Neural.Data.Basic;
using Encog.Neural.Rbf.Training;
using Encog.Neural.RBF;

namespace rbf
{
    class Program
    {
        static void Main(string[] args)
        {
            RbfTester rbfTester = new RbfTester();
            for(int type = 0; type < 5; type++)
            {
                for (int i = 10; i < 101; i += 10)
                {
                    rbfTester.run(i, 1000, type, true, RBFEnum.MexicanHat);
                    rbfTester.run(i, 1000, type, false, RBFEnum.MexicanHat);
                    rbfTester.run(i, 1000, type, true, RBFEnum.Gaussian);
                    rbfTester.run(i, 1000, type, false, RBFEnum.Gaussian);
                    rbfTester.run(i, 1000, type, true, RBFEnum.InverseMultiquadric);
                    rbfTester.run(i, 1000, type, false, RBFEnum.InverseMultiquadric);
                    rbfTester.run(i, 1000, type, true, RBFEnum.Multiquadric);
                    rbfTester.run(i, 1000, type, false, RBFEnum.Multiquadric);
                }
                Console.ReadLine();
            }
            
            Console.Out.Write("\nPress any key to continue...");
            Console.ReadLine();
        }
    }
}
