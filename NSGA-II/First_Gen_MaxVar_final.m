%===============================================================
%                                                               %
%  MATLAB Code for Stepwise Opt.                                %
%  Non-dominated Sorting Genetic Algorithm II (NSGA-II)         %
%                                                               %
%                                                               %
%  Sejong Univ. K.-S. Sohn                                      %
%                                                               %
%         e-Mail: kssohn@sejong.ac.kr                           %
%         M.P:  010-6253-5913                                   %
%                                                               %
%        First Generation Maker                                 %
%===============================================================
clc;
clear;
close all;

npop=10;                                    % main population size
nvar=4;                                     % number of unknown variables
nobj=numel(FittnessFunction(zeros(1,nvar)));    % number of objective functions

maxit=100;                      % maximum number of iterations

%VarMin=-5;                      % lower bound of unknown variables
%VarMax= 5;                      % upper bound of unknown variables
%VarRange=[VarMin VarMax];       % range of unknown variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VarMin= [0.5 0.5 0.5 0.5];                % x1 (1~41),  x2 (1~85), x3 (1~2), x4 (1~3)
VarMax= [41.5 85.5 2.5 3.5];               

pc=0.8;                         % crossover ratio
nc=round(pc*npop/2)*2;          % number of parents (also offsprings)

pm=0.3;                         % mutation ratio
nm=round(pm*npop);              % number of mutants

% initialization
pop=CreateEmptyIndividuals(npop);
for i=1:numel(pop)
    var=zeros(1, nvar);
    for j = 1:nvar
    var(j)=unifrnd(VarMin(j),VarMax(j),[1 1]);
    end
    pop(i).Position=var;
    
   y=pop(i).Position;

    
   % pop(i).Cost=FittnessFunction(pop(i).Position);
    y2=pop(i).Position;
    y3=pop(i).Position;
    y4=pop(i).Position;
    y5=pop(i).Position;
    y6=pop(i).Position;
    y7=pop(i).Position;
    y8=pop(i).Position;
    y9=pop(i).Position;
    y10=pop(i).Position;
    y11=pop(i).Position;
    y12=pop(i).Position;
    y13=pop(i).Position;
    y14=pop(i).Position;
    y15=pop(i).Position;
    y16=pop(i).Position;
    y17=pop(i).Position;
    y18=pop(i).Position;
    y19=pop(i).Position;
    y20=pop(i).Position;
    
    
    
aver = 0;
for ii = 1 : nvar
y(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver = aver + (y(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver = aver / nvar;
Var1 = 0;
for j = 1: nvar
Var1 = Var1 + (aver - (y(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end
    
aver2 = 0;
for ii = 1 : nvar
y(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver2 = aver2 + (y2(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver2 = aver2 / nvar;
Var2 = 0;
for j = 1: nvar
Var2 = Var2 + (aver2 - (y2(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var2 > Var1
    Var1 = Var2;
for iii = 1 : nvar-1
y(iii) = y2(iii);
end
end


aver3 = 0;
for ii = 1 : nvar-1
y3(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver3 = aver3 + (y3(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver3 = aver3 /  (nvar-1);
Var3 = 0;
for j = 1: nvar-1
Var3 = Var3 + (aver3 - (y3(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var3 > Var1
    Var1 = Var3;
for iii = 1 : nvar-1
y(iii) = y3(iii);
end
end


aver4 = 0;
for ii = 1 : nvar-1
y4(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver4 = aver4 + (y4(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver4 = aver4 /  (nvar-1);
Var4 = 0;
for j = 1: nvar-1
Var4 = Var4 + (aver4 - (y4(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var4 > Var1
    Var1 = Var4;
for iii = 1 : nvar-1
y(iii) = y4(iii);
end
end


aver5 = 0;
for ii = 1 : nvar-1
y5(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver5 = aver5 + (y5(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver5 = aver5 /  (nvar-1);
Var5 = 0;
for j = 1: nvar-1
Var5 = Var5 + (aver5 - (y5(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var5 > Var1
    Var1 = Var5;
for iii = 1 : nvar-1
y(iii) = y5(iii);
end
end

aver6 = 0;
for ii = 1 : nvar-1
y6(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver6 = aver6 + (y6(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver6 = aver6 /  (nvar-1);
Var6 = 0;
for j = 1: nvar-1
Var6 = Var6 + (aver6 - (y6(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var6 > Var1
    Var1 = Var6;
for iii = 1 : nvar-1
y(iii) = y6(iii);
end
end


aver7= 0;
for ii = 1 : nvar-1
y7(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver7 = aver7 + (y7(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver7= aver7 /  (nvar-1);
Var7 = 0;
for j = 1: nvar-1
Var7 = Var7 + (aver7 - (y7(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var7 > Var1
    Var1 = Var7;
for iii = 1 : nvar-1
y(iii) = y7(iii);
end
end


aver8 = 0;
for ii = 1 : nvar-1
y8(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver8 = aver8 + (y8(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver8 = aver8 /  (nvar-1);
Var8 = 0;
for j = 1: nvar-1
Var8 = Var8 + (aver8 - (y8(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var8 > Var1
    Var1 = Var8;
for iii = 1 : nvar-1
y(iii) = y8(iii);
end
end


aver9 = 0;
for ii = 1 : nvar-1
y9(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver9 = aver9 + (y9(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver9 = aver9 / (nvar-1);
Var9 = 0;
for j = 1: nvar-1
Var9 = Var9 + (aver9 - (y9(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var9 > Var1
    Var1 = Var9;
for iii = 1 : nvar-1
y(iii) = y9(iii);
end
end

aver10 = 0;
for ii = 1 : nvar-1
y10(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver10 = aver10 + (y10(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver10 = aver10 / (nvar-1);
Var10 = 0;
for j = 1: nvar-1
Var10 = Var10 + (aver10 - (y10(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var10 > Var1
    Var1 = Var10;
for iii = 1 : nvar-1
y(iii) = y10(iii);
end
end


aver11 = 0;
for ii = 1 : nvar-1
y11(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver11 = aver11 + (y11(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver11 = aver11 / (nvar-1);
Var11 = 0;
for j = 1: nvar-1
Var11 = Var11 + (aver11 - (y11(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var11 > Var1
    Var1 = Var11;
for iii = 1 : nvar-1
y(iii) = y11(iii);
end
end


aver12 = 0;
for ii = 1 : nvar-1
y12(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver12 = aver12 + (y12(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver12 = aver12 / (nvar-1);
Var12 = 0;
for j = 1: nvar-1
Var12 = Var12 + (aver12 - (y12(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var12 > Var1
    Var1 = Var12;
for iii = 1 : nvar-1
y(iii) = y12(iii);
end
end


aver13 = 0;
for ii = 1 : nvar-1
y13(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver13 = aver13 + (y13(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver13 = aver13 / (nvar-1);
Var13 = 0;
for j = 1: nvar-1
Var13 = Var13 + (aver13 - (y13(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var13 > Var1
    Var1 = Var13;
for iii = 1 : nvar-1
y(iii) = y13(iii);
end
end


aver14 = 0;
for ii = 1 : nvar-1
y14(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver14 = aver14 + (y14(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver14 = aver14 / (nvar-1);
Var14 = 0;
for j = 1: nvar-1
Var14 = Var14 + (aver14 - (y14(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var14 > Var1
    Var1 = Var14;
for iii = 1 : nvar-1
y(iii) = y14(iii);
end
end


aver15 = 0;
for ii = 1 : nvar-1
y15(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver15 = aver15 + (y15(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver15 = aver15 / (nvar-1);
Var15 = 0;
for j = 1: nvar-1
Var15 = Var15 + (aver15 - (y15(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var15 > Var1
    Var1 = Var15;
for iii = 1 : nvar-1
y(iii) = y15(iii);
end
end


aver16 = 0;
for ii = 1 : nvar-1
y16(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver16 = aver16 +(y16(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver16 = aver16 / (nvar-1);
Var16 = 0;
for j = 1: nvar-1
Var16 = Var16 + (aver16 - (y16(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var16 > Var1
    Var1 = Var16;
for iii = 1 : nvar-1
y(iii) = y16(iii);
end
end

aver17 = 0;
for ii = 1 : nvar-1
y17(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver17 = aver17 + (y17(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver17 = aver17 / (nvar-1);
Var17 = 0;
for j = 1: nvar-1
Var17 = Var17 + (aver17 - (y17(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var17 > Var1
    Var1 = Var17;
for iii = 1 : nvar-1
y(iii) = y17(iii);
end
end


aver18 = 0;
for ii = 1 : nvar-1
y18(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver18 = aver18 + (y18(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver18 = aver18 / (nvar-1);
Var18 = 0;
for j = 1: nvar-1
Var18 = Var18 + (aver18 - (y18(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var18 > Var1
    Var1 = Var18;
for iii = 1 : nvar-1
y(iii) = y18(iii);
end
end


aver19 = 0;
for ii = 1 : nvar-1
y19(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver19 = aver19 + (y19(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver19 = aver19 / (nvar-1);
Var19 = 0;
for j = 1: nvar-1
Var19 = Var19 + (aver19 - (y19(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var19 > Var1
    Var1 = Var19;
for iii = 1 : nvar-1
y(iii) = y19(iii);
end
end


aver20 = 0;
for ii = 1 : nvar-1
y20(ii) = unifrnd(VarMin(ii),VarMax(ii),[1 1]);
aver20 = aver20 + (y20(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii));
end
aver20 = aver20 / (nvar-1);
Var20 = 0;
for j = 1: nvar-1
Var20 = Var20 + (aver20 - (y20(ii)-VarMin(ii))/(VarMax(ii)-VarMin(ii))) ^ 2;
end

if Var20 > Var1
    Var1 = Var20;
for iii = 1 : nvar
y(iii) = y20(iii);
end
end

    
     % pop(i).Position=y;

   % pop(i).Cost=FittnessFunction(pop(i).Position);

  % R
   var=zeros(1, nvar);
    for j = 1:4
    var(j)=round(y(j));
    end
    for j = 5:nvar
    var(j)=y(j);
    end
    pop(i).Position=var;
    %End
end

   pitP=reshape([pop.Position],nvar,[]);

          
  % pitC=reshape([pop.Cost],nobj,[]) ;
   
   pitPC=[[pitP]' ]';
  
       fgP=fopen(sprintf('%d.txt',1),'w');
       fprintf(fgP, '%g %g %g %g \n', [pitPC]);
       fclose(fgP);
