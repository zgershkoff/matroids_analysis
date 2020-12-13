import sage.all
from sage.matroids.constructor import Matroid
from sage.matrix.constructor import Matrix
from sage.rings.finite_rings.finite_field_constructor import GF
import sage.matroids.matroids_catalog as matroids
from sage.graphs.graph_generators import graphs
import json

def string_to_matrix(repstring, rank = 0):
    """
    interprets a string of space-separated integers as a binary matrix

    INPUT:  string as from Gordon Royle's binary matroid data, and the
            rank of the matrioid.
    OUTPUT: binary matrix

    The input strings are the number of elements followed by decimal
    representations of the binary vectors. For example, the string
    "4 1 2 4 7 " is a rank-3 4-element matroid given by
    [[0,0,1], [0,1,0], [1,0,0], [1,1,1]].
    Although the rank is given as input because the data are organized
    by rank, it can be surmised from the number of bits needed for the
    last number.
    """
    elts = [int(n) for n in repstring.strip().split(" ")[1:]]
    mat = []
    if not rank:
        rank = elts[-1].bit_length()
    for n in elts:
        mat.append([int(c) for c in format(n, '0' + str(rank) + 'b')])
        # alternate solution: bin(int(n))[2:].zfill(7)
    return Matrix(GF(2), mat).transpose()

def matrix_to_matroid(mat):
    """
    returns the matroid of a given matrix

    INPUT:   A matrix.
    OUPTUT:  A matroid.
    """
    return Matroid(mat)

# These next two methods could be considered stubs.
# Enumerating circuits and cocircuits is computationally hard, and
# there's not a clean way to start with the smallest ones.
# A different approach could be a binary search on the possible ranks,
# but this isn't the right thing to focus on right now.
# And besides, it's NP-hard (A. Vardy, The intractability of computing
# the minimum distance of a code)
def girth(M, circuits):
    """
    returns the size of the smallest circuit

    INPUT:   A matroid and its set of circuits
    OUTPUT:  Integer
    """
    return min(len(C) for C in circuits)

def cogirth(M, circuits):
    """
    returns the size of the smallest cocircuit

    INPUT:   A matroid and its set of circuits
    OUTPUT:  Integer
    """
    if len(M.coloops()) > 0:
        return 1
    elif not M.is_cosimple():
        return 2
    return min(len(C) for C in circuits)

def has_triangle(M, circuits):
    """
    returns true if M has a triangle
    INPUT:   A matroid and its set of circuits
    OUTPUT:  Integer
    """
    return any(len(C) == 3 for C in circuits)

def analyze_matroid(M):
    """
    checks parameters of a matroid

    INPUT:   A matroid
    OUTPUT:  A list consisting of booleans and integers

    The attributes measured are:
    - Fano minor and dual
    - K_{3,3} minor and dual
    - K_5 minor and dual
    - Triangle minor
    - Cosimple
    - 2, 3, and 4-connected
    - Girth and cogirth
    """
    attribs = [False] * 11 # 11 boolean variables
    attribs.extend([0, 0]) # girth and cogirth

    # checking for minors can be improved with a database approach
    attribs[0] = M.has_minor(matroids.named_matroids.Fano())
    attribs[1] = M.has_minor(matroids.named_matroids.Fano().dual())
    attribs[2] = M.has_minor(Matroid(graphs.CompleteBipartiteGraph(3,3)))
    attribs[3] = M.has_minor(matroids.named_matroids.K33dual())
    attribs[4] = M.has_minor(matroids.CompleteGraphic(5))
    attribs[5] = M.has_minor(matroids.CompleteGraphic(5).dual())

    circuits = M.circuits() # this is the slow step
    attribs[6] = has_triangle(M, circuits)
    attribs[7] = M.is_cosimple()
    attribs[8] = M.is_connected()
    attribs[9] = M.is_3connected()
    attribs[10] = M.is_4connected()

    attribs[11] = girth(M, circuits)
    attribs[12] = cogirth(M, circuits)

    return attribs

def main():
    InFile = "../data/smallBinaryMatroids/hr-sz13-rk05"
    OutFile = "../json/hr-sz13-rk05-results3.json"
    rank = 5

    result = []

    with open(InFile, 'r') as f:
        for repstring in f:
            mat = string_to_matrix(repstring, rank)
            M = matrix_to_matroid(mat)
            attribs = analyze_matroid(M)
            result.append((repstring, attribs))

    with open(OutFile, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
