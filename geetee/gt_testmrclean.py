import gtIO as io
import gtMRClean as clean

def main():
    
    bad_frame = '../Data/BadFrame_2.png'
    
    # Read bad frame
    fr = io.LoadImage(bad_frame, border=16)
    
    # Repair frame
    fr_clean, artifact = clean.MRClean(fr, verbose=True)
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

