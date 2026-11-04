# Does the 22 dB PSNR floor still earn its keep?

Shipped default `Tokenizer()` (text-tuned DReX, S1 refiner), same image, `floor=22` vs `floor=0`. 
Tokens are counted tokens (`Tokenizer.length`), not packed length.

- 10/419 (image, side) pairs change budget when the floor is on (2.4%).
- On those, median tokens 784 -> 2304, median PSNR 19.41 -> 22.20 dB, median LPIPS 0.4338 -> 0.1383.
- Encode time overall: median 123 ms -> 139 ms with the floor (+12%).

## side 768

| group | n | floor active | tokens nf -> floor | PSNR nf -> floor | LPIPS nf -> floor | enc ms nf -> floor |
|---|---|---|---|---|---|---|
| dense | 9 | 0 | 1521 -> 1521 | 30.16 -> 30.16 | 0.0637 -> 0.0637 | 162 -> 162 |
| doclaynet | 90 | 0 | 2304 -> 2304 | 32.53 -> 32.53 | 0.0049 -> 0.0049 | 229 -> 228 |
| funsd | 90 | 0 | 2304 -> 2304 | 32.98 -> 32.98 | 0.0053 -> 0.0053 | 232 -> 231 |
| mondrian | 13 | 0 | 75 -> 75 | 33.83 -> 33.83 | 0.1678 -> 0.1678 | 91 -> 92 |
| nonaxis | 16 | 0 | 27 -> 27 | 30.59 -> 30.59 | 0.1290 -> 0.1290 | 91 -> 92 |
| ornate | 6 | 0 | 961 -> 961 | 35.38 -> 35.38 | 0.0369 -> 0.0369 | 119 -> 118 |
| rot00 | 13 | 1 | 47 -> 47 | 33.95 -> 33.95 | 0.1856 -> 0.1856 | 90 -> 91 |
| rot15 | 13 | 1 | 60 -> 60 | 31.07 -> 31.07 | 0.2212 -> 0.1975 | 88 -> 89 |
| rot30 | 13 | 1 | 60 -> 60 | 30.32 -> 30.32 | 0.1973 -> 0.1805 | 89 -> 89 |
| rot45 | 13 | 1 | 60 -> 60 | 29.83 -> 29.83 | 0.1839 -> 0.1720 | 91 -> 92 |
| simple | 8 | 1 | 234 -> 234 | 30.26 -> 30.26 | 0.1765 -> 0.1765 | 96 -> 96 |
| synthcanvas | 8 | 0 | 75 -> 75 | 33.73 -> 33.73 | 0.1860 -> 0.1860 | 91 -> 91 |
| synthclean | 8 | 0 | 51 -> 51 | 35.20 -> 35.20 | 0.0227 -> 0.0227 | 88 -> 88 |
| synthdog | 90 | 0 | 729 -> 729 | 34.93 -> 34.93 | 0.0260 -> 0.0260 | 110 -> 112 |
| synthgrain | 8 | 0 | 51 -> 51 | 38.55 -> 38.55 | 0.0178 -> 0.0178 | 88 -> 89 |
| text | 5 | 3 | 784 -> 1444 | 20.55 -> 22.88 | 0.1029 -> 0.0922 | 104 -> 190 |
| texture | 13 | 2 | 729 -> 841 | 23.64 -> 23.64 | 0.2093 -> 0.1810 | 108 -> 113 |
| waldo | 3 | 0 | 841 -> 841 | 27.69 -> 27.69 | 0.1056 -> 0.1056 | 108 -> 106 |

## Floor-active cases, worst 10 by PSNR gained

| image | side | tokens | PSNR nf -> floor | LPIPS nf -> floor | psnr calls |
|---|---|---|---|---|---|
| file:texture/04_Gravel_road_with_loose_gravel_close.jpg | 768 | 538 -> 2304 | 15.42 -> 22.20 | 0.1938 -> 0.0419 | 4 |
| file:text/text_medium_sans.jpg | 768 | 784 -> 1521 | 20.55 -> 25.25 | 0.0894 -> 0.0351 | 2 |
| file:simple/29_mj_tensorian_machine_and_bi.jpg | 768 | 409 -> 784 | 20.49 -> 24.10 | 0.1684 -> 0.1442 | 2 |
| file:text/text_small_serif.jpg | 768 | 961 -> 1849 | 18.86 -> 22.36 | 0.3869 -> 0.0922 | 2 |
| file:text/text_picture_serif.jpg | 768 | 625 -> 1225 | 19.58 -> 22.80 | 0.3058 -> 0.1383 | 2 |
| file:texture/25_Dülmen,_Hausdülmen,_Felder_am_Müh.jpg | 768 | 784 -> 2304 | 18.14 -> 20.69 | 0.6628 -> 0.0801 | 3 |
| file:rot15/04_Theo_van_Doesburg_Composition_IX.jpg | 768 | 900 -> 2304 | 19.41 -> 20.66 | 0.4338 -> 0.1362 | 3 |
| file:rot45/04_Theo_van_Doesburg_Composition_IX.jpg | 768 | 784 -> 2304 | 19.30 -> 20.53 | 0.4755 -> 0.1398 | 3 |
| file:rot30/04_Theo_van_Doesburg_Composition_IX.jpg | 768 | 784 -> 2304 | 19.04 -> 19.94 | 0.4346 -> 0.1590 | 3 |
| file:rot00/04_Theo_van_Doesburg_Composition_IX.jpg | 768 | 900 -> 2304 | 20.20 -> 20.29 | 0.4468 -> 0.2170 | 3 |
