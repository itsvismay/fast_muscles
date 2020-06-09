t0 = t(1);
t1 = t(end);
f0 = min(f);
f1 = max(f);
h = plot(t,f);
fig = gcf;
fig.Units = "inches";
fig.PaperUnits = "inches";
fig.PaperPosition = [0 0 4 2];
fig.Position = [0 0 4 2];
h.LineWidth = 2;
h.Color = [0,0,0]/255;
% h.Marker = "o";
% h.MarkerFaceColor = [0,0,0]/255;;
% h.MarkerEdgeColor = [0 0 0];
a = gca;
% t0 = floor(min(t));
% t1 = ceil(max(t)/5)*5;
% 
% f0 = 0*floor(min(f));
% f1 = ceil(max(f)/3)*3;
a.XLim = [t0, t1];
a.YLim = [0, 3];
%a.YLim = [0, 1e4];
a.GridColor = [0 0 0]/255;
a.Color = [229 229 229]/255;
a.XTick = [0, 10, 20, 30, 40];
a.YTick = [0,1,2,3];
a.YGrid = "On";
a.GridColor = [1 1 1];
a.GridAlpha = 1;
a.Box = "off";
ylabel("Distance from FEM");
xlabel("Alpha Value");
title ("Relationship Between Alpha and Distance From FEM"); 
a.FontName = "Linux Biolinum O";
a.FontSize = 8.25;
% a.XScale = "log";
%a.YScale = "log";
a.XMinorTick = "off";
a.YMinorTick = "off";
%a.XTick = "off"
%a.YTick = "off"; 
a.XMinorGrid = 'off';
a.YMinorGrid = "off"
hold on

h2 = plot(t,f2);
h2.LineWidth = 2;
h2.Color = [0,0,0]/255
% h2.Marker = "o";
% h2.MarkerFaceColor = [211,53,43]/255
% h2.MarkerEdgeColor = [0 0 0];
hold on

h3 = plot(t,f3);
h3.LineWidth = 2;
h3.Color = [0,0,0]/255
% h3.Marker = "o";
% h3.MarkerFaceColor = [211,53,43]/255
% h3.MarkerEdgeColor = [0 0 0];
hold on

h4 = plot(t,f4);
h4.LineWidth = 2;
h4.Color = [0,0,0]/255
% h4.Marker = "o";
% h4.MarkerFaceColor = [211,53,43]/255
% h4.MarkerEdgeColor = [0 0 0];
hold on

h5 = plot(t,f5);
h5.LineWidth = 2;
h5.Color = [0,0,0]/255
% h5.Marker = "o";
% h5.MarkerFaceColor = [211,53,43]/255
% h5.MarkerEdgeColor = [0 0 0];
hold on

h6 = plot(t,f6);
h6.LineWidth = 2;
h6.Color = [0,0,0]/255
% h6.Marker = "o";
% h6.MarkerFaceColor = [211,53,43]/255
% h6.MarkerEdgeColor = [0 0 0];
hold on

h7 = plot(t,f7);
h7.LineWidth = 2;
h7.Color = [0,0,0]/255
% h7.Marker = "o";
% h7.MarkerFaceColor = [211,53,43]/255
% h7.MarkerEdgeColor = [0 0 0];
hold on
