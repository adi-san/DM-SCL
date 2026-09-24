from helper_functions import *

V_prime, n_a_prime, n_b_prime, F_out, C_A, C_B, r, F_in, C_A0, V, n_a, n_b, k_c, V_sp = sp.symbols('V_prime n_a_prime n_b_prime F_out C_A C_B r F_in C_A0 V n_a n_b k_c V_sp')

equations = [
    sp.Eq(V_prime, F_in-F_out),
    sp.Eq(n_a_prime, F_in*C_A0-F_out*C_A-V*r),
    sp.Eq(n_b_prime, -F_out*C_B+V*r),
    sp.Eq(C_A, n_a/V),
    sp.Eq(C_B, n_b/V),
    sp.Eq(F_out,k_c*(V-V_sp)+F_in),
]
variables = [V_prime, n_a_prime, n_b_prime, F_out, C_A, C_B, r]

G, M, var_order = build_bipartite_graph(equations, variables)

print("Variable order:", var_order)
print("Incidence matrix:\n", M.toarray())
print("Graph edges:", list(G.edges()))

# do dulmage_mendelsohn decomposition

row_partition, col_partition = dulmage_mendelsohn(M)

print("Square rows (well-constrained):", row_partition.square)
print("Overconstrained rows:", row_partition.overconstrained)
print("Underconstrained rows:", row_partition.underconstrained)
print("Unmatched rows:", row_partition.unmatched)
print("Square columns (well-constrained):", col_partition.square)
print("Overconstrained columns:", col_partition.overconstrained)
print("Underconstrained columns:", col_partition.underconstrained)
print("Unmatched columns:", col_partition.unmatched)

# I want to also get a list of the matched pairs, including in overdetermined and underdetermined blocks
matched_pairs = []
for r_block, c_block in [
    (row_partition.square, col_partition.square),
    (row_partition.overconstrained, col_partition.overconstrained),
    (row_partition.underconstrained, col_partition.underconstrained)
]:
    matched_pairs.extend(zip(r_block, c_block))

print("Matched pairs (row, col):", matched_pairs)

# take process to SS, eliminate derivative vars (V_prime, n_a_prime, n_b_prime) from incidence matrix as well as last 3 equations
# eliminate columns with indices 0,1 and 2 as well as last 3 rows
M_ss = M.copy()
M_ss = M_ss.toarray()
M_ss = np.delete(M_ss, [0, 1, 2], axis=1)  # remove columns 0, 1, 2
M_ss = np.delete(M_ss, [-3, -2, -1], axis=0)  # remove last 3 rows
# now measure F_out --> delete the corresponding column
M_ss = np.delete(M_ss, [0], axis=1)  # assuming F_out is now the first column after previous deletions
print("Incidence matrix at SS (without derivatives and F_out):\n", M_ss)
# do DM Decomposition on the steady-state incidence matrix
M_ss=scipy.sparse.coo_matrix(M_ss)
row_partition_ss, col_partition_ss = dulmage_mendelsohn(M_ss)

print("Square rows (well-constrained) at SS:", row_partition_ss.square)
print("Overconstrained rows at SS:", row_partition_ss.overconstrained)
print("Underconstrained rows at SS:", row_partition_ss.underconstrained)
print("Unmatched rows at SS:", row_partition_ss.unmatched)
print("Square columns (well-constrained) at SS:", col_partition_ss.square)
print("Overconstrained columns at SS:", col_partition_ss.overconstrained)
print("Underconstrained columns at SS:", col_partition_ss.underconstrained)
print("Unmatched columns at SS:", col_partition_ss.unmatched)

# get matched pairs at steady state
matched_pairs_ss = []
for r_block, c_block in [
    (row_partition_ss.square, col_partition_ss.square),
    (row_partition_ss.overconstrained, col_partition_ss.overconstrained),
    (row_partition_ss.underconstrained, col_partition_ss.underconstrained)
]:
    matched_pairs_ss.extend(zip(r_block, c_block))

print("Matched pairs at SS (row, col):", matched_pairs_ss)

# Now eliminate C_A from M_ss
M_ss = M_ss.toarray()
M_ss = np.delete(M_ss, [0], axis=1)  # assuming C_A is now the first column after previous deletions
M_ss=scipy.sparse.coo_matrix(M_ss)
row_partition_ss, col_partition_ss = dulmage_mendelsohn(M_ss)

print("Square rows (well-constrained) at SS after eliminating C_A:", row_partition_ss.square)
print("Overconstrained rows at SS after eliminating C_A:", row_partition_ss.overconstrained)
print("Underconstrained rows at SS after eliminating C_A:", row_partition_ss.underconstrained)
print("Unmatched rows at SS after eliminating C_A:", row_partition_ss.unmatched)
print("Square columns (well-constrained) at SS after eliminating C_A:", col_partition_ss.square)
print("Overconstrained columns at SS after eliminating C_A:", col_partition_ss.overconstrained)
print("Underconstrained columns at SS after eliminating C_A:", col_partition_ss.underconstrained)
print("Unmatched columns at SS after eliminating C_A:", col_partition_ss.unmatched)
# get matched pairs at steady state after eliminating C_A
matched_pairs_ss_after_C_A = []
for r_block, c_block in [
    (row_partition_ss.square, col_partition_ss.square),
    (row_partition_ss.overconstrained, col_partition_ss.overconstrained),
    (row_partition_ss.underconstrained, col_partition_ss.underconstrained)
]:
    matched_pairs_ss_after_C_A.extend(zip(r_block, c_block))

print("Matched pairs at SS after eliminating C_A (row, col):", matched_pairs_ss_after_C_A)

# this code is rather repetitive --> write a function that takes in an incidence matrix and then outputs the
# DM decomposition results and matched pairs
def analyze_incidence_matrix(M):
    M = M.toarray() if not isinstance(M, np.ndarray) else M
    M = scipy.sparse.coo_matrix(M)
    row_partition, col_partition = dulmage_mendelsohn(M)

    print("Square rows (well-constrained):", row_partition.square)
    print("Overconstrained rows:", row_partition.overconstrained)
    print("Underconstrained rows:", row_partition.underconstrained)
    print("Unmatched rows:", row_partition.unmatched)
    print("Square columns (well-constrained):", col_partition.square)
    print("Overconstrained columns:", col_partition.overconstrained)
    print("Underconstrained columns:", col_partition.underconstrained)
    print("Unmatched columns:", col_partition.unmatched)

    matched_pairs = []
    for r_block, c_block in [
        (row_partition.square, col_partition.square),
        (row_partition.overconstrained, col_partition.overconstrained),
        (row_partition.underconstrained, col_partition.underconstrained)
    ]:
        matched_pairs.extend(zip(r_block, c_block))

    print("Matched pairs (row, col):", matched_pairs)
    return row_partition, col_partition, matched_pairs