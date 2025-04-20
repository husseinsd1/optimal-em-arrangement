import numpy as np
import cvxpy as cp


def find_opt_positions(M, K, U_matrix, poses, em_length):
    x = cp.Variable(shape=(M))
    t = cp.Variable(1)

    UUT = construct_UUT(U_matrix)

    alpha = 0.1
    tol = 1.0e-4
    obj = cp.Maximize(t)
    cons1 = x >= 0.0
    cons2 = x <= 1.0  
    cons3 = cp.sum(x) == K 
    cons4 = t <= cp.lambda_min(UUT_operator(x, UUT, M))
    constraints = [cons1, cons2, cons3, cons4]

    # Mutex constraint
    for o in overlapping(poses, em_length):
        constraints.append(cp.sum([x[j] for j in o]) <= 0.8)

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    print("Status: ", prob.status)
    print("Lowest Eigen Val: ", t.value)
    return x.value


def UUT_operator(x, UUT, M):
    return cp.sum([x[i] * UUT[i] for i in range(M)], axis=0)


def construct_UUT(U_matrix):
    return np.array([np.outer(U_matrix[i], U_matrix[i]) for i in range(U_matrix.shape[0])])


def overlapping(poses, em_length):
    overlapping = []
    for i in range(len(poses)):
        for j in range(i+1, len(poses)):
            if np.linalg.norm(poses[i][0] - poses[j][0]) < 1.5 * em_length:
                overlapping.append([i, j])
    return overlapping