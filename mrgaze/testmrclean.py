import mrgaze.io as mrio
import mrgaze.mrclean as mrm

def main():
    
    bad_frame = '../Data/BadFrame_2.png'
    
    # Read bad frame
    fr = mrio.LoadImage(bad_frame, border=16)
    
    # Repair frame
    fr_clean, artifact = mrm.MRClean(fr, verbose=True)
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

