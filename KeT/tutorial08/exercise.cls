\ProvidesClass{exercise}
\LoadClass[11pt,a4paper]{scrartcl}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{braket}
\usepackage[
  math-style=ISO,    % ┐
  bold-style=ISO,    % │
  sans-style=italic, % │ Follow ISO standard
  nabla=upright,     % │
  partial=upright,   % ┘
  warnings-off={           % ┐
    mathtools-colon,       % │ Suppress unnecessary
    mathtools-overbracket, % │ warnings
  }                        % ┘
]{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{ifthen}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{wrapfig}
\usepackage{polyglossia}
\ifx\languageEN\undefined
  \setmainlanguage{german}
\else
  \setmainlanguage{english}
\fi
\usepackage[autostyle]{csquotes}

\ifx\languageEN\undefined
  \usepackage[
    locale=DE,
    separate-uncertainty=true,
    per-mode=symbol-or-fraction,
    input-decimal-markers={.},
    output-decimal-marker={,},
  ]{siunitx}
\else
  \usepackage[
    locale=UK,
    separate-uncertainty=true,
    per-mode=symbol-or-fraction,
    input-decimal-markers={.},
    output-decimal-marker={.},
  ]{siunitx}
\fi

\usepackage{epsfig}
\usepackage{type1cm}
\usepackage{icomma}
\hypersetup{
  colorlinks=false,
  pdfcreator=pdf-LaTeX
}
\areaset{170mm}{250mm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1.5ex plus0.5ex minus0.5ex}
\pagestyle{plain}

\newcommand{\eventDE}[1]{\def\EventDE{#1}}
\newcommand{\eventEN}[1]{\def\EventEN{#1}}
\newcommand{\lecturer}[2][]{\def\Lecturer{\hfill#1\\\hfill#2}}
\newcommand{\submission}[1]{\def\Submission{#1}}
\newcommand{\release}[1]{\def\Release{#1}}
\newcommand{\semester}[1]{\def\Semester{#1}}
\renewcommand{\labelenumi}{\bfseries\alph{enumi})}
\renewcommand{\labelenumii}{\roman{enumii})}

\newcounter{tutorial}
\newcounter{exercise}

\newcommand{\exercise}[1]{
  \vspace*{5mm}
  \ifx\languageEN\undefined
    {\bfseries\large\underline{Aufgabe \theexercise:}}
  \else
    {\bfseries\large\underline{Exercise \theexercise:}}
  \fi
  \stepcounter{exercise}{\large #1} \\ [2mm]
}

\newcommand{\solution}[1]{
  \ifx\modelsolution\undefined
  \else
    \ifx\languageEN\undefined
      \par{\bfseries\em Lösung: } #1
    \else
      \par{\bfseries\em Solution: } #1
    \fi
  \fi
}

\newcommand{\notsolution}[1]{
  \ifx\modelsolution\undefined
    \ifx\languageEN\undefined
      #1
    \else
      #1
    \fi
  \else
  \fi
}

\newcommand{\makeheader}{
  \noindent{
    \begin{center}
      {
        \fontfamily{cmss}\selectfont\huge{\arabic{tutorial}}.
        \ifx\languageEN\undefined
          \EventDE
        \else
          \EventEN
        \fi
        \hfill\Semester \\
      }
      \vspace{1em}
      \ifx\languageEN\undefined
        \begin{tabular}{ll}
          \textbf{Ausgabe:} & \Release \\
          \textbf{Abgabe:}  & \Submission
        \end{tabular}
      \else
        \begin{tabular}{ll}
          \textbf{Release date:} & \Release \\
          \textbf{Due:}          & \Submission
        \end{tabular}
      \fi
      \hfill
      \begin{tabular}{r}
        \Lecturer
      \end{tabular}
    \end{center}
    \hrule
    \vspace{24pt}
  }
}