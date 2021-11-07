# Design of a Graph Neural Network to predict the optimal resolution of the Sonar Performance Model.
Master of Science Thesis by Jakub Pietrak

_Faculty of Mechanical, Maritime and Materials Engineering (3mE) · Delft University of
Technology_

## Foreword
This repository is an archive of the code written for my graduation project carried out at TNO Acoustic & Sonar in Den Haag. 
Its main purpose is the presentation of replicable results and the outline of the proposed methods. Attached descriptions focus on the code functionality and hence they rarely dive into mathematical derivations or the decision process. For those interested, the complete report can be found [here](https://github.com/kubpie/SPM-Thesis/blob/master/docs/mscThesis.pdf "MSc Thesis Report").

## Project Introduction
**Graph Neural Networks** are a unique type of Deep Learning models that have a capability to exploit an explicitly stated structure of data representation. By design they carry a strong _relational inductive bias_, which is a set of assumptions that makes the algorithm prioritize some solutions over another, independent of observed data. This makes the method especially interesting for applications to problems that are naturally relation-centric, or in which local interactions between features are the main value of interest.

<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/problem_setup.JPG" alt="drawing" width="700"/>
</p>

The presented research case, aims to explore GNN potential in application to an Ocean Acoustics problem. Using the geometric ray-tracing algorithm - BELLHOP - a number of underwater sound propagation scenarios (aprox. 40,000 data points) were simulated. Each scenario is described by a set of simulation input parameters and a Sound Speed Profile (SSP) function. SSP has a critical impact on sound propagation mode, acting as a guideline for reflected and refracted paths of rays travelling through a water column. A predictive model to effectively capture acoustic phenomena, requires a mean of representing interactions in very scarce feature space with respect to the unknown polynomial function representation of the sound speed profile. 

## General Problem Approach



