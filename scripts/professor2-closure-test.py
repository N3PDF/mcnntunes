import os, sys
import numpy as np

def parseArgs(argv):
   import argparse
   parser = argparse.ArgumentParser(prog='Prof2-closure-test', description='''
   This program performs a closure test of the tuning procedure: for each MC run
   in the data folder, it performs the Professor tuning procedure using the MC run as
   the experimental data, and then compares the fitted parameters with the ones used during
   the run generation.
   ''')
   parser.add_argument("-d", "--data", help="Folder with benchmark data", required=True, type=str)
   parser.add_argument("-r", "--runs", help="Folder with MC runs", required=True, type=str)
   parser.add_argument("--order", help="The order of the polynomial interpolation", required=True, type=int)
   parser.add_argument("-w", "--wfile", help="Weightfile", required=True, type=str)
   parser.add_argument("-o", "--output", help="Temporary output for the tunes", required=True, type=str)
   parser.add_argument("--results", help="Output for the benchmark results", required=True, type=str)
   parser.add_argument("--tune_options", help="Additional options for prof2-tune", default='', type=str)
   return parser.parse_args(argv)

def main(argv):

    args = parseArgs(argv)
    f = open(args.results,'w')

    # Parametrizing the histograms
    os.system("prof2-ipol "+args.runs+" --order="+str(args.order))
    
    # Looking for the benchmark data in the --data folder
    foldernames = os.listdir(args.data)

    # Preparing the accumulators
    chi2 = np.zeros(len(foldernames))
    mean_relative_difference = np.zeros(len(foldernames))

    for i, foldername in enumerate(foldernames):

        # Reading the true parameters
        true_params = {}
        with open(args.data+foldername+"/params.dat",'r') as param_f:
            for line in param_f.readlines():
                (k, v) = line.split()
                true_params[k.strip()] = v.strip()
        
        # Tuning with Professor
        os.system("prof2-tune -d "+args.data+foldername+" -r "+args.runs+" --wfile "+args.wfile+" -o "+args.output+" "+args.tune_options+" ipol.dat")

        # Reading the tuned parameters
        with open(args.output+"/results.txt",'r') as param_f:
            lines = param_f.readlines()
        params = {}
        for key in true_params:
            params[key] = [] 
        for line in lines:
            line = line.replace('\n','')
            line = line.replace('#','')
            for key in params:
                if key in line:
                    try: # If there are more than two value it isn't the value nor the error
                        (name, value) = line.split()
                    except:
                        continue
                    params[key].append(value.strip())

        # Calculating an estimator
        truth = np.zeros(len(true_params))
        predicted = np.zeros(len(true_params))
        errors = np.zeros(len(true_params))
        for j, key in enumerate(true_params):
            truth[j] = true_params[key]
            for key2 in params:
                if key == key2:
                    predicted[j] = params[key][0]
                    errors[j] = params[key][1]   
        chi2[i]=np.mean(np.square((truth - predicted)/errors))
        mean_relative_difference[i] = np.mean(np.abs((predicted - truth)/truth))*100
        
        # Printing a comparison
        f.write("##################################################\n")
        f.write("Results of the closure test on benchmark run "+args.data+foldername+":\n")
        f.write("Weightfile: %s \n" % args.wfile)
        f.write("{0:20}\t\t{1:20}\t\t{2:20}\t\t{3:20}\n".format("Params","True value","Predicted value","Error"))
        for key in true_params:
            for key2 in params:
                if key == key2:
                    f.write("{0:20}\t\t{1:20}\t\t{2:20}\t\t{3:20}\n".format(key,true_params[key],params[key][0],params[key][1]))
        f.write("Average chi2/dof: %f\n" % chi2[i])
        f.write("Average relative difference: %f %%\n" % mean_relative_difference[i])

    # Saving overall results
    f.write("##################################################\n")
    f.write("Total average chi2: %f +- %f \n" %
            (np.mean(chi2), np.sqrt(np.var(chi2) / chi2.shape[0])))
    f.write("Total average relative difference: %f %% +- %f %% \n" %
            (np.mean(mean_relative_difference), np.sqrt(np.var(mean_relative_difference) / mean_relative_difference.shape[0])))

    f.close()

if __name__=="__main__":
   main(sys.argv[1:])