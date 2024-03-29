# The Rolling Hough Transform (RHT)

This is the Rolling Hough Transform, described in [Clark, Peek, & Putman 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...789...82C/abstract), ApJ 789, 82 (arXiv:1312.1338). If use of the RHT results in a publication, please cite this work.

For instructions on install, use, and more, please see http://seclark.github.io/RHT/.

The RHT is written and maintained by Susan E Clark. Please feel free to get in touch with questions or submit pull requests with improvements.

Update 2/2021, S.E. Clark: I recently pushed convRHT.py. This is a version of the RHT with the core architecture rewritten to be based on a series of convolutions. It is much faster than the original RHT. This should eventually be integrated into the main code base, but I am releasing it for now so that more people can use it. If you use convRHT in a publication, please cite the main RHT paper as usual, as well as BICEP/Keck Collaboration et al. 2022 (arXiv:2210.05684), corresponding author George Halal, where the convRHT is first introduced. As always, feel free to reach out with comments and questions, or submit pull requests. Many thanks to convRHT beta-testers G. Halal and Y.K. Ma. 

Update 8/2023, S.E. Clark: For a version of the RHT that runs directly on the sphere (HealPix), see [Halal, Clark et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230610107H/abstract).
