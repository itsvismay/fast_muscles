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
h.Color = [102, 172, 87]/255;
h.Marker = "o";
h.MarkerFaceColor = [102, 172, 87]/255;;
h.MarkerEdgeColor = [0 0 0];
a = gca;
% t0 = floor(min(t));
% t1 = ceil(max(t)/5)*5;
% 
% f0 = 0*floor(min(f));
% f1 = ceil(max(f)/3)*3;
a.XLim = [t0, t1];
a.YLim = [0, 200];
%a.YLim = [0, 1e4];
a.GridColor = [0 0 0]/255;
a.Color = [229 229 229]/255;
a.XTick = [1, 4, 8, 12];
a.YTick = [0, 50, 100, 150, 200];
a.YGrid = "On";
a.GridColor = [1 1 1];
a.GridAlpha = 1;
a.Box = "off";
ylabel("Time (s)");
xlabel("# of Cores");
title ("Multi-Threaded Scaling of EMu"); 
a.FontName = "Linux Biolinum O";
a.FontSize = 8.25;
%a.XScale = ?log?;
%a.YScale = ?log?;
a.XMinorTick = ?off?;
a.YMinorTick = ?off?;
%a.XTick = ?off?
%a.YTick = ?off?; 
a.XMinorGrid = ?off?;
a.YMinorGrid = ?off?
hold on
h2 = plot(t,f2);
h2.LineWidth = 2;
h2.Color = [211,53,43]/255
h2.Marker = ?o?;
h2.MarkerFaceColor = [211,53,43]/255
h2.MarkerEdgeColor = [0 0 0];
hold on
h3 = plot(t,f3);
h3.LineWidth = 2;
h3.Color = [211,53,43]/255
h3.Marker = ?o?;
h3.MarkerFaceColor = [211,53,43]/255
h3.MarkerEdgeColor = [0 0 0];
%range of plot 3 cells, 4 numbers round top and bottom to nearest 5