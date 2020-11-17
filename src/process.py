import sage.all
from sage.matroids.constructor import Matroid
from sage.matrix.constructor import Matrix
from sage.rings.finite_rings.finite_field_constructor import GF
import sage.matroids.matroids_catalog as matroids
from sage.graphs.graph_generators import graphs
import json

def string_to_matrix(repstring, rank):
    """
    INPUT: string as from Gordon Royle's binary matroid data
    OUTPUT: binary matrix
    """
    elts = repstring.strip().split(" ")[1:]
    mat = []
    for n in elts:
        # Todo: adjust according to rank
        mat.append([int(c) for c in format(int(n), '0' + str(rank) + 'b')])
        # alternate solution: bin(int(n))[2:].zfill(7)
    return Matrix(GF(2), mat).transpose()

def matrix_to_matroid(mat):
    return Matroid(mat)

# these are stubs
# improve speed and add checks for matroids with rank or corank 0
def girth(M):
    return min(len(C) for C in M.circuits())

def cogirth(M):
    return min(len(C) for C in M.cocircuits())

def analyze_matroid(M):
    attribs = [False] * 12 # 12 boolean variables
    attribs.extend([0, 0]) # girth and cogirth

    # checking for minors can be improved with a database approach
    attribs[0] = M.has_minor(matroids.named_matroids.Fano())
    attribs[1] = M.has_minor(matroids.named_matroids.Fano().dual())
    attribs[2] = M.has_minor(Matroid(graphs.CompleteBipartiteGraph(3,3)))
    attribs[3] = M.has_minor(matroids.named_matroids.K33dual())
    attribs[4] = M.has_minor(matroids.CompleteGraphic(5))
    attribs[5] = M.has_minor(matroids.CompleteGraphic(5).dual())

    attribs[6] = M.has_line_minor(3)
    attribs[7] = M.is_simple()
    attribs[8] = M.is_cosimple()
    attribs[9] = M.is_connected()
    attribs[10] = M.is_3connected()
    attribs[11] = M.is_4connected()

    attribs[12] = girth(M)
    attribs[13] = cogirth(M)

    return attribs

def main():
    InFile = "../data/smallBinaryMatroids/hr-sz13-rk05"
    OutFile = "../json/hr-sz13-rk05-results.json"
    rank = 5

    result = []

    with open(InFile, 'r') as f:
        for repstring in f:
            mat = string_to_matrix(repstring, rank)
            M = matrix_to_matroid(mat)
            attribs = analyze_matroid(M)
            result.append([repstring] + [str(a) for a in attribs])

    with open(OutFile, 'w') as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    main()
