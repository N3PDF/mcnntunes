# -*- coding: utf-8 -*-
"""
Build theory predictions from MC in order to use mcnntunes.
"""
import glob
import sys, os
import yoda

def main():
    # Parse argument
    args = parseArgs()

    # Read the string @xmin:xmax and save it in the patts list!
    NeedToChangeExpData=False
    patts=[]
    for patt in args.patterns:
        tmp=yoda.search.PointMatcher(patt)
        patts.append(tmp)
        if tmp.index != None:
            NeedToChangeExpData=True

    # Iterate over number of runs (MC data)
    for run in range(args.num):
        Save(args, args.dir+os.path.sep+str(run).zfill(4)+os.path.sep+args.yodafile, patts, run=run)
        
    
    if NeedToChangeExpData == True:
        
        if args.expData == None:
            print('"--expData" required')
            sys.exit(0)

        yodafiles=[]
        for dataPath in args.expData:
            for patt in patts:
                path=patt.path              # re.compile(path)
                path=path.pattern           # get the string path from re.compile()
                path=path.split('/')[0]     # Get thd first part CMS_...._........
                
                yodafile=glob.glob(os.path.join(dataPath,'**',path)+'.yoda', recursive=True)    # search for the match in expData paths (ex: ./Rivet/) RECURSIVELY!
                try:
                    yodafile=yodafile[0]    # If is more than one take the first one!
                except:
                    pass
                yodafiles.append(yodafile)
        
        # Iterate on the expData
        for yodafile in yodafiles:
            Save(args, yodafile, patts, expData=True)
    

def Save(args, yodafile, patts, run=None, expData=False):
    # Read input files
    aos = yoda.read(yodafile)
    if expData==False:
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
        
        
        # Read the conditions
        MyScatter=None

        save_pat=True       # if no args.patterns ---> Save all
        save_unpat=False     # is no args.unpatterns ---> Save all
        
        if args.unpatterns != None and expData==False:      # Enter if unpatterns
            for unpattern in args.unpatterns:               # loop on args.unpatterns
                if unpattern in scatterplot.path():         # check if unpath match scatterplot.path
                    save_unpat=True
        
        if patts != None:
            save_pat=False
            for patt in patts:
                path=patt.path
                path=path.pattern
                if path in scatterplot.path().replace('/REF',''):
                    save_pat=True
                    MyScatter=ListOfPoints(patt,scatterplot,aopath, ao)
                    MyScatter=MyScatter.myScatter
                    if expData==False:
                        MyScatter.setAnnotation("Run_Directory", args.dir)
                        for key,value in param.items():
                            MyScatter.setAnnotation("Tune_Parameter_"+key, value)
        if save_pat==True and save_unpat==False:
            scatters.append(MyScatter)
    
    # Create folder and write YODA file
    nameDataFolder='experimental_data'
    if not os.path.exists(args.output) and expData==False:
        os.makedirs(args.output)
    elif not os.path.exists(nameDataFolder) and expData==True:
        os.makedirs(nameDataFolder)
    if expData==False:    
        yoda.writeYODA(scatters, os.path.join(args.output, "run"+str(run).zfill(4)+"_"+args.yodafile))
    elif expData==True:
        head, tail = os.path.split(yodafile)
        yoda.writeYODA(scatters, os.path.join(nameDataFolder, tail))


# check if points (center of the bin) in range @xmin:xmax
class ListOfPoints(object):
    def __init__(self, patt, scatterplot, aopath, ao) :
        pointToSave=[]
        n_bins=0
        for p in scatterplot.points():
            point=POINT(p)
            check=patt.match_pos(point)
            if check==True:
                n_bins += 1
                pointToSave.append(p)
        self.myScatter=yoda.core.Scatter2D(pointToSave,aopath, ao.title())
    
# This class is given in input to check if the point p is in the range @xmin:xmax
class POINT(object):
    def __init__(self, p):
        self.xmax=p.x()
        self.xmin=p.x()

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
    parser.add_argument("--expData", nargs='+', help='paths to rivet analysis (experimental data)', required=False)
    return parser.parse_args()

if __name__ == "__main__":
    main()