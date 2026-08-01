# vendor/

Checkout or symlink [`earth-genome/majortom-rs`](https://github.com/earth-genome/majortom-rs)
here as `majortom-rs` so the Cargo path dependency resolves:

```bash
mkdir -p vendor
ln -sfn ../../majortom-rs vendor/majortom-rs
```

CI clones the repo into this path automatically. The `majortom-rs` directory itself
is gitignored.
