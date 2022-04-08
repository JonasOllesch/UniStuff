%   Programmierblatt 1

% Lukas Annuss, Jonas Ollesch

close all;  % Löscht alte Plots
clear all; 

% Definition der Funktionen
f = @funktion;


% Polynom vierten Grades
n = 4;%... Grad 4
x4 = linspace(-5,5,n+1);%... äquidistante Knoten mittels linspace
y4 = f(x4);%... Daten
p4 = polyfit(x4,y4,n);%... Interpolationspolynom mittels polyfit		

% Polynom zehnten Grades
n = 10;
x10 = linspace(-5,5,n+1);
y10 = f(x10);
p10 = polyfit(x10,y10,n);		

%% Auswertung
xx = linspace(-5,5,101); % diskrete Definitionsmenge
yyf = f(xx);%... Auswertung von f an den Stellen xx
yy4 = polyval(p4,xx);%... Auswertung von p4 an den Stellen xx mittels polyval	
yy10 = polyval(p10,xx);%... Auswertung von p10 an den Stellen xx mithilfe polyval
r4 = abs(f(xx) - yy4);%... Interpolationsfehler zu p4
r10 = abs(f(xx) - yy10);%...Interpolationsfehler zu p10

%% Ausgabe
subplot(3,1,1); % Aufteilung des Plot-Fensters in drei Zeilen
title('Funktion f, Daten für x und y sowie Interpolationspolynom vom Grad 4') % Überschrift des Plots
hold on % Damit alle Plots auf einmal auftauchen?
plot(xx,yyf,"red"); % Plot von f(x)
plot(x4,y4,"green"); % Die Knoten und Daten
plot(xx, yy4,"blue"); % Interpolationspolynom vom Grad 4
legend('f(x)','Daten','Interpolationspolynom vom Grad 4'); % Labeln der Plots
hold on

subplot(3,1,2);
title('Funktion f, Daten für x und y sowie Interpolationspolynom vom Grad 10') % analog zu Grad 4
hold on
plot(xx,yyf,"red");
plot(x10,y10,"green");
plot(xx, yy10,"blue");
legend('f(x)','Daten','Interpolationspolynom vom Grad 10');
hold on

subplot(3,1,3);
title('Fehlerfunktionen')
hold on
plot(xx,r4); % Plot des Fehlers für p4
plot(xx,r10); % Fehlerplot für p10
legend('Fehler für Interpolation vom Grad 4','Fehler für Interpolation vom Grad 10'); % Beschriftung
hold on

function a = funktion(x) % Definition von f(x)
a = 10./(1+x.^2);
end