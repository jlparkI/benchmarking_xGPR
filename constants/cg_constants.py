"""Parameters for use in testing conjugate gradients with and without
preconditioning."""
import numpy as np

fitting_rffs = [16384, 32768]
precond_rank = [0, 256, 512, 1024]
preset_hyperparams = [np.asarray([-1.4701868288258673,1.488503077974549,
                                        -0.5559783514308828]),
                    np.asarray([-0.18500497156080836,-0.9688407884695313,
                                        -2.3766176234444494]),
                    np.asarray([-0.41483738691418204,-0.19847253218546643,
                                        0.45796802733361996]),
                    np.asarray([-0.7477801295755524,2.3003848691265802,
                                        -2.9805562399929806]),
                    np.asarray([-0.8832280464765179,0.23252013335197366,
                                        -0.7564112993375892]),
                    np.asarray([-1.0572382378575833,0.17987230132295984,
                                        -0.6770193344307492]),
                    np.asarray([-0.6440528332353815,0.09055420482101785,
                                        -1.6778624979265484]),
                    np.asarray([-0.7100270148315362,1.3174664066993527,
                                        -2.412056958370786])   ]

fitcomp_preset_hyperparams = [np.asarray([-1.4701868288258673,1.488503077974549,
                                        -0.5559783514308828]),
                    np.asarray([-0.18500497156080836,-0.9688407884695313,
                                        -2.3766176234444494]),
                    np.asarray([-0.41483738691418204,-0.19847253218546643,
                                        0.45796802733361996]),
                    np.asarray([-0.6440528332353815,0.09055420482101785,
                                        -1.6778624979265484])
                    ]

fitcomp_max_rank = [1600, 1024, 1024, 1024]
