{-# LANGUAGE BangPatterns #-}
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.BLAS

import Prelude hiding (fst, snd)

main :: IO ()
main = do
    putStrLn "-- Testing dot product"
    putStrLn "  < (1, .., 10), (0, .., 9) > = "
    print $ run1 f (v1, v2)
    putStrLn "  < (1, .., 10000), (0, .., 9999) > = "
    print $ run1 g (w1, w2)

    putStrLn "-- Testing a.x + y"
    putStrLn "  0.5*(1, .., 10) + (0, .., 9)"
    print $ run1 h (half1, v1, v2)
    putStrLn "  2.1*(1, .., 10000) + (0, .., 9999)"
    print $ run1 h' (half2, w1, w2)

    putStrLn "-- Testing asum"
    putStrLn "  asum (1, .., 10)"
    print $ run1 sasum v1
    putStrLn "  asum (0, .., 9999)"
    print $ run1 dasum w2

    putStrLn "-- Testing norm2"
    putStrLn "  nrm2 (1, .., 10)"
    print $ run1 snrm2 v1
    putStrLn "  nrm2 (0, .., 9999)"
    print $ run1 dnrm2 w2

    putStrLn "-- Testing scal"
    putStrLn "  scal 0.5 (1, .., 10)"
    print $ run1 sc (half1, v1)
    putStrLn "  scal 10 (1, .., 30)"
    print $ run1 sc' (half2, w3)

  where v1, v2 :: Array DIM1 Float
        !v1   = fromList (Z :. 10) [1..10]
        !v2   = fromList (Z :. 10) [0..9]

        w1, w2 :: Array DIM1 Double
        !w1   = fromList (Z :. 10000) [1..10000]
        !w2   = fromList (Z :. 10000) [0..9999]
        !w3   = fromList (Z :. 30)    [1..30]

        half1 :: Scalar Float
        !half1 = fromList Z [0.5]

        half2 :: Scalar Double
        !half2 = fromList Z [2]

f :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
f vs = let (v1, v2) = (fst vs, snd vs) :: (Acc (Vector Float), Acc (Vector Float))
       in v1 `sdot` v2

g :: Acc (Vector Double, Vector Double) -> Acc (Scalar Double)
g vs = let (v1, v2) = (fst vs, snd vs) :: (Acc (Vector Double), Acc (Vector Double))
       in v1 `ddot` v2

h :: Acc (Scalar Float, Vector Float, Vector Float) -> Acc (Vector Float)
h as = let (s, v1, v2) = unlift as
       in saxpy s v1 v2

h' :: Acc (Scalar Double, Vector Double, Vector Double) -> Acc (Vector Double)
h' as = let (s, v1, v2) = unlift as
        in daxpy s v1 v2

sc :: Acc (Scalar Float, Vector Float) -> Acc (Vector Float)
sc vs = let (scalar, vec) = unlift vs
        in sscal scalar vec

sc' :: Acc (Scalar Double, Vector Double) -> Acc (Vector Double)
sc' vs = let (scalar, vec) = unlift vs
         in dscal scalar vec
