# Generate the one-hot encoding dictionary
ONEHOT_EMBEDDINGS = {i: [1 if j == i - 1 else 0 for j in range(100)] for i in range(1, 101)}

# Save the dictionary to a file
with open("onehot_embeddings.py", "w") as f:
    f.write("ONEHOT_EMBEDDINGS = {\n")
    for key, value in ONEHOT_EMBEDDINGS.items():
        f.write(f"    {key}: {value},\n")
    f.write("}\n")
