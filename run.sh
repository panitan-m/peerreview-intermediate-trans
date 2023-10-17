# # Intermediate Task training
for aspect in clarity meaningful_comparison motivation originality soundness substance
do
    python train_intermediate.py --aspect $aspect \
    --out_dir asap_models/$aspect
done

# # Target Task Fine-tuning
for asap in clarity meaningful_comparison motivation originality soundness substance
do
    for aspect in clarity meaningful_comparison impact originality recommendation soundness_correctness substance
    do
        for seed in 1234 2345 3456
        do
            for e in {1..10}
            do
                python finetune.py \
                --out_dir results/$seed/$aspect/${asap}_i/e$e \
                --aspects $aspect \
                --checkpoint asap_models/$asap/checkpoint.e$e.pth.tar \
                --seed $seed
            done
        done
    done
done
