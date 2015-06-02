function LBP_CreatePositives(Positive_Dir)
% Create positive info file for opencv_createsamples
% 
% AUTHOR : Mike Tyszka
% PLACE  : Caltech
% DATES  : 2014-05-12 JMT From scratch

if nargin < 1; Positive_Dir = pwd; end

% Loop over all image files contained in Positive_Dir
d = dir(Positive_Dir);

% Create image figure for defining ROIs
figure(1); clf; colormap(gray)

% Open positive info file (see opencv_createsamples documentation)
fd = fopen('Positive.txt','w');

for dc = 1:length(d)
    
    this_file = d(dc).name;
    
    fprintf('Image : %s\n', this_file);
    
    [~, ~, fext] = fileparts(this_file);
    
    if isequal(fext,'.jpg') || isequal(fext,'.png')
        
        this_image = fullfile(Positive_Dir, this_file);
        
        % Load image
        s = imread(this_image);

        % Convert to grayscale if necessary
        if size(s,3) == 3
            s = rgb2gray(s);
        end
        
        % Robust intensity correction
        s = imadjust(s);
        
        % Display image
        imshow(s);
        
        % Init continuation flag
        keep_going = true;
        
        % Clear any current ROI info
        p1 = [];
        
        while keep_going
        
            k = waitforbuttonpress;
            
            if k == 0
                
                % Refresh image before drawing new ROI
                imshow(s)
            
                point1 = get(gca,'CurrentPoint');    % button down detected
                
                rbbox;
                
                point2 = get(gca,'CurrentPoint');    % button up detected
                point1 = point1(1,1:2);              % extract x and y
                point2 = point2(1,1:2);
                
                p1 = fix(min(point1,point2));        % calculate locations
                offset = fix(abs(point1-point2));    % and dimensions
                
                x = [p1(1) p1(1)+offset(1) p1(1)+offset(1) p1(1) p1(1)];
                y = [p1(2) p1(2) p1(2)+offset(2) p1(2)+offset(2) p1(2)];
                
                % Overlay ROI on image
                hold on
                axis manual
                plot(x,y,'w','linewidth',2)
                hold off
                
            else
               
                % Key was pressed - get from frame
                ch = get(gcf, 'CurrentCharacter');
                
                switch lower(ch)
                    
                    case 'a' % Rotate 90 CCW
                        fprintf('Rotate -90\n');
                        s = rot90(s,1);
                        
                    case 'd' % Rotate 90 CW
                        fprintf('Rotate +90\n');
                        s = rot90(s,-1);
                        
                    case 's' % Save ROI and adjusted image then move on

                        if isempty(p1)
                            
                            fprintf('No ROI defined\n');
                            
                            % Leave keep_going as true and continue
                            
                        else
                        
                            keep_going = false;
                        
                            % Report and save ROI bounds
                            fprintf('Saving ROI bounds : (%d %d %d %d)\n', p1(1), p1(2), offset(1), offset(2));
                            fprintf(fd, '%s 1 %d %d %d %d\n', this_image, p1(1), p1(2), offset(1), offset(2));
                        
                            % Save adjusted image
                            fprintf('Resaving adjusted image\n');
                            imwrite(s, this_image);
                            
                        end
                        
                    case 'q' % Quit
                        fprintf('Quitting\n');
                        close all
                        return
                        
                    otherwise
                        % Do nothin
                        
                end
                
                % Refresh display
                imshow(s)
                
            end
                
        end % Image rotation and ROI loop

    end
    
end

fprintf('Done\n');

% Clean up
close all
fclose(fd);
    
