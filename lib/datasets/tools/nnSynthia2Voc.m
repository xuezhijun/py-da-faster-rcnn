% Convert Synthia dataset to VOC format
% Image size = 1280x760
% HB 2017
%%
close all;
clear;
clc;

synthiaPath = fullfile('PATH_TO_RAND_CITYSCAPES');
currentSet = 'VOC2007';
destDir = fullfile('PATH_TO_SYNTHIA-VOC');
if ~exist(destDir,'dir')
    mkdir(destDir);
end

%%
cmap = jet(6);
labelMap(1).name = 'car';
labelMap(1).id = 8;
labelMap(1).pascalId = 1;
labelMap(1).color = cmap(1,:);
labelMap(2).name = 'bus';
labelMap(2).id = 19;
labelMap(2).pascalId = 2;
labelMap(2).color = cmap(2,:);
labelMap(3).name = 'motorcycle';
labelMap(3).id = 12;
labelMap(3).pascalId = 3;
labelMap(3).color = cmap(3,:);
labelMap(4).name = 'bicycle';
labelMap(4).id = 11;
labelMap(4).pascalId = 4;
labelMap(4).color = cmap(4,:);
labelMap(5).name = 'person';
labelMap(5).id = 10;
labelMap(5).pascalId = 5;
labelMap(5).color = cmap(5,:);
labelMap(6).name = 'rider';
labelMap(6).id = 17;
labelMap(6).pascalId = 6;
labelMap(6).color = cmap(6,:);

% Create a label map file
fName = fullfile(destDir,sprintf('synthia_label_map.pbtxt'));
fid = fopen(fName,'w');
fprintf(fid,'item {\n   id: 0\n   name: ''backgroud, always 0 in VOC format''\n}\n\n');
for i=1:length(labelMap)
    fprintf(fid,'item {\n   id: %i\n   name: ''%s''\n}\n\n',labelMap(i).pascalId,lower(labelMap(i).name));
end
fclose(fid);

%%
xVal = {'trainval'};
for m=1:length(xVal)
    fName = fullfile(destDir,xVal{m},currentSet,'Annotations');
    mkdir(fName);
    fName = fullfile(destDir,xVal{m},currentSet,'ImageSets','Main');
    mkdir(fName);
    fName = fullfile(destDir,xVal{m},currentSet,'JPEGImages');
    mkdir(fName);
end

%%
% Prepare files listing which image contains which class
fids = cell(length(xVal),length(labelMap)); 
for x=1:length(xVal)
for v=1:length(labelMap)
    fName = fullfile(destDir,xVal{x},currentSet,'ImageSets','Main',sprintf('%s_%s.txt',lower(labelMap(v).name),xVal{x}));
    fids{x,v} = fopen(fName,'w');
end
end

%%
files = dir(fullfile(synthiaPath,'RGB','*.png'));
cntr = 0;
i = 1;
while cntr<(9400)
    mode = 'trainval';
    outputFileName = sprintf('%s',files(i).name(1:7));
    outputJpegFileName = sprintf('%s',files(i).name);
    outputXmlFileName = sprintf('%s.xml',files(i).name(1:7));
    imageName = fullfile(synthiaPath,'RGB',files(i).name);
    labelsName = fullfile(synthiaPath,'GT','LABELS',files(i).name);
    
    %image = imread(imageName);
    %imageSize = [size(image,1) size(image,2)];
    imageSize = [760 1280];
    labels = imread(labelsName);
    objectLabels = labels(:,:,1);
    objectInstances = labels(:,:,2);
    objects = getBndBox(objectLabels,objectInstances,labelMap);
    if isempty(objects)
        continue;
    end

%%
    annotation.folder = currentSet;
    annotation.filename = sprintf('%s',files(i).name);
    annotation.source.annotation = fullfile('GT','LABELS',files(i).name);
    annotation.source.database = 'SYNTHIA-RAND';
    annotation.source.image = files(i).name;
    
    %annotation.size.depth = size(image,3);
    %annotation.size.height = size(image,1);
    %annotation.size.width = size(image,2);
    annotation.size.depth = 3;
    annotation.size.height = 760;
    annotation.size.width = 1280;
    
    annotation.object = objects(:);
    
    %imwrite(image,fullfile(destDir,mode,currentSet,'JPEGImages',outputJpegFileName));
    s.annotation = annotation;
    struct2xml(s,fullfile(destDir,mode,currentSet,'Annotations',outputXmlFileName));
    
    sel = cellfun(@(x) strcmp(x,mode),xVal);
    for c=1:length(labelMap)
       for o=1:length(annotation.object)
            if strcmpi(labelMap(c).name,annotation.object{o}.name)
                %isPresent = strcmpi(annotation.object{o}.name,labelMap(c).name)*2-1;
                %fprintf(fids{sel,c},'%s %i\n',outputFileName,isPresent);
                % We don't care if multiple objects are present, so we break out.
                fprintf(fids{sel,c},'%s\n',outputFileName);
                break;
            end
        end
    end
    i = i+1;
    cntr = cntr+1;
end

%%
% Close files
for i=1:numel(fids)
    fclose(fids{i});
end