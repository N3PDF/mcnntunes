# -*- coding: utf-8 -*-
"""
Build theory predictions from MC in order to use mcnntunes.
"""
import sys, os
import yoda

def main():

    # Parse argument
    args = parseArgs()

    # Iterate over number of runs
    for run in range(args.num):

        # Read input files
        aos = yoda.read(args.dir+os.path.sep+str(run).zfill(4)+os.path.sep+args.yodafile)
        fpar = open(args.dir+os.path.sep+str(run).zfill(4)+os.path.sep+args.paramfile,'r')
        param = {}
        for line in fpar:
            (k, v) = line.split()
            param[k.strip()] = v.strip()
        fpar.close()

        # Extract scatterplots from yoda file
        scatters = []
        for aopath, ao in aos.items():
            scatterplot = ao.mkScatter()
            scatterplot.setAnnotation("Run_Directory", args.dir)
            for key,value in param.items():
                scatterplot.setAnnotation("Tune_Parameter_"+key, value)

            # Check for patterns and unpatterns
            save=True
            if args.patterns != None:
                save=False
                for pattern in args.patterns:
                    if pattern in scatterplot.path():
                        save=True         
            if args.unpatterns != None:
                for unpattern in args.unpatterns:
                    if unpattern in scatterplot.path():
                        save=False
            if save:
                scatters.append((aopath, scatterplot))

        # Create folder and write YODA file
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        yoda.writeYODA(dict(scatters), os.path.join(args.output, "run"+str(run).zfill(4)+"_"+args.yodafile))

def parseArgs():
    """Parse argument"""
    import argparse
    parser = argparse.ArgumentParser(description='Builds theory prediction from MC runs')
    parser.add_argument("-n", "--num", help="number of runs", required=True, type=int)
    parser.add_argument("-d", "--dir", help="absolute path to folder containing runs subfolders", required=True)
    parser.add_argument("-f", "--yodafile", help="YODA file name", required=True)
    parser.add_argument("-p", "--paramfile", help="parameter file name", default="params.dat")
    parser.add_argument("--patterns", nargs='+', help="list of patterns to extract", required=False)
    parser.add_argument("--unpatterns", nargs='+', help="list of patterns to avoid", required=False)
    parser.add_argument("-o", "--output", help="output folder (default: output)", default="output")
    return parser.parse_args()

if __name__ == "__main__":
    main()

