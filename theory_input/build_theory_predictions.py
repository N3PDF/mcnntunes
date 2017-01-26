#! /usr/bin/env python
##

import sys,os
import yoda

def parseArgs(argv):
   import argparse
   parser = argparse.ArgumentParser(description='Builds theory prediction from MC runs')
   parser.add_argument("-n", "--num", metavar='N', help="number of runs", required='True', type=int)
   parser.add_argument("-d", "--dir", help="Absolute path to folder containing runs subfolders", required='True')
   parser.add_argument("-f", "--yodafile", help="YODA file name", required='True')
   parser.add_argument("-p", "--paramfile", help="parameter file name", required='True')
   return parser.parse_args(argv)


def main(argv):
   args = parseArgs(argv)
   for run in range(args.num):
       aos = yoda.read(args.dir+os.path.sep+str(run).zfill(4)+os.path.sep+args.yodafile)
       fpar = open(args.dir+os.path.sep+str(run).zfill(4)+os.path.sep+args.paramfile,'r')
       param = {}
       for line in fpar:
          (k, v) = line.split()
          param[k.strip()] = v.strip()
       fpar.close()
       for aopath, ao in aos.iteritems():
           ao.setAnnotation("Run_Directory",args.dir)
           for key,value in param.iteritems():
              ao.setAnnotation("Tune_Parameter_"+key, value)
       yoda.writeYODA(aos,"run"+str(run).zfill(4)+"_"+args.yodafile)


####################
if __name__=="__main__":
   main(sys.argv[1:])
