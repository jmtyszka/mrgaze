import os
from mrgaze import media, mrclean

def main():
    
    bad_frame = '/Users/jmt/GitHub/mrgaze/mrgaze/Data/BadFrame_2.png'
    
    if not os.path.isfile(bad_frame):
        print('* Image not found - exiting')
        return False
    
    # Read bad frame
    fr = media.LoadImage(bad_frame, border=16)
    
    # Repair frame
    fr_clean, artifact = mrclean.MRClean(fr, 5.0)
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

