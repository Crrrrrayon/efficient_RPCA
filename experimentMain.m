clc
clear
%% load dataset
imagePath=fullfile('.\dataset\IBMtest2\') ;
temp=dir(imagePath); 
fileNames={temp.name};
hiddenFiles=nan(size(fileNames));
for fileIndex=1:length(fileNames)
    tempName=fileNames{fileIndex};
    hiddenFiles(fileIndex)=tempName(1) == '.';
end
names=fileNames(~hiddenFiles);

inputImg=imread(fullfile(imagePath,names{1}));
shape=size(inputImg);
R_channel=[];
G_channel=[];
B_channel=[];

for i=1:length(names)
    name=names{i};
    imageName=fullfile(imagePath,name);
    image=imread(imageName);
    
    Red=image(:,:,1);
    Green=image(:,:,2);
    Blue=image(:,:,3);
    
    TR=convertMat2Vector(Red);
    R_channel(:,i)=TR;   
    TG=convertMat2Vector(Green);
    G_channel(:,i)=TG;
    TB=convertMat2Vector(Blue);
    B_channel(:,i)=TB;
end
tol = 1e-7;

%% eRPCA
para.beta_init = 1.5*max(abs(R_channel(:)));
para.beta      = para.beta_init;
para.tol       = 1e-7;
para.con       = 5;
para.eta       = 0.7;
tic
[AR,ER] = eRPCA(R_channel,para);
[AG,EG] = eRPCA(G_channel,para);
[AB,EB] = eRPCA(B_channel,para);
toc

%% save results
for i=1:length(names)
    LR=AR(:,i);
    LG=AG(:,i);
    LB=AB(:,i);
    
    SR=ER(:,i);
    SG=EG(:,i);
    SB=EB(:,i);

    %%low rank structure
    LR_shape=convertVector2Mat(LR,shape(1),shape(2));
    LG_shape=convertVector2Mat(LG,shape(1),shape(2));
    LB_shape=convertVector2Mat(LB,shape(1),shape(2));
    L=cat(3,LR_shape,LG_shape,LB_shape);
    L=uint8(L);
    
    %%Sparse part
    SR_shape=convertVector2Mat(SR,shape(1),shape(2));
    SG_shape=convertVector2Mat(SG,shape(1),shape(2));
    SB_shape=convertVector2Mat(SB,shape(1),shape(2));  
    S=cat(3,SR_shape,SG_shape,SB_shape);
    S=uint8(abs(S));
    imwrite(L,['.\Result\IBMtest2\L\',num2str(i,'%05d'),'.jpg']);
    imwrite(S,['.\Result\IBMtest2\S\',num2str(i,'%05d'),'.jpg']);
end
