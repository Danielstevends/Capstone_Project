# Generator Machine Learning Emission Modeling
### Created by: Daniel Sitompul 
#### UC Berkeley - Renewable and Appropriate Energy Laboratory (RAEL)

This model is created to assess the use of machine learning to predict generator ussage within a time period. The data that we have is voltage and frequency (within a 2 minute period) from 2 locations in the healthcare facilities.

In this repository, I use 3 different machine learning model:
1. Logistic Regression
2. Random Forest
3. XG-Boost

More information about the methodology and result could be depicted in the presentation file: Presentation - Climate Impact Assessment of Generator Usage using Machine Learning_ A Case Study from DR Congo.pdf

\section*{Prepare Environment}

To prepare the environment, follow these steps:

\subsection*{1. Create a Virtual Environment}
\begin{verbatim}
python -m venv venv
\end{verbatim}

\subsection*{2. Activate the Virtual Environment}
- On Windows:
\begin{verbatim}
venv\Scripts\activate
\end{verbatim}
- On macOS/Linux:
\begin{verbatim}
source venv/bin/activate
\end{verbatim}

\subsection*{3. Install Dependencies}
Install the required libraries:
\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

\subsection*{4. Verify Installation}
Make sure all dependencies are installed:
\begin{verbatim}
pip freeze
\end{verbatim}


