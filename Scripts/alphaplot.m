%alpha plots
f0 = [0.5867, 
    0.5866
    0.5865,
    0.5853,
    0.576,
    0.573,
    0.557,
    0.52,
    0.474,
    0.408,
    0.345,
    0.093];

f1 = [0.5587,
    0.5587,
    0.55846,
    0.5557,
    0.549,
    0.553,
    0.532,
    0.487,
    0.437,
    0.4065,
    0.304,
    0.1327];
t = linspace(1, size(f0,1), size(f0,1));

h1 = plot(t, f0);
h1.LineWidth = 5;
h1.Color = [82, 91, 210]/255;
h1.Marker = ".";
h1.MarkerSize = 30;
h1.MarkerFaceColor = [82, 91, 210]/255
h1.MarkerEdgeColor = [82, 91, 210]/255;
hold on
  
h2 = plot(t, f1);
h2.LineWidth = 5;
h2.Color = [102, 179, 73]/255;
h2.Marker = ".";
h2.MarkerSize = 30;
h2.MarkerFaceColor = [102, 179, 73]/255
h2.MarkerEdgeColor = [102, 179, 73]/255;
hold on

fig = gcf;
fig.Units = "inches";
fig.PaperUnits = "inches";
fig.PaperPosition = [0 0 4 2];
fig.Position = [0 0 4 2];

a = gca;
a.XLim = [min(t), max(t)];
a.YLim = [min([f0;f1]), max([f0;f1])];
a.GridColor = [0 0 0]/255;
a.Color = [229 229 229]/255;
% a.XScale = "log";
% a.YScale = "log";
a.XTick = [min(t), max(t)];
a.YTick = [min([f0;f1]), max([f0;f1])];
a.YGrid = "On";
a.GridColor = [1 1 1];
a.GridAlpha = .1;
a.Box = "off";
ylabel("Eigenvalue");
xlabel("Eigenvalue #");
title ("Inverse Eigenspectrum of ");
a.FontSize = 20;
a.FontWeight = 'bold';