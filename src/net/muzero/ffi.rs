use libloading::Library;

#[allow(clippy::missing_safety_doc)]
pub unsafe fn load_tricked_ops() -> Option<Library> {
    let lib_paths = [
        "tricked_ops.so",
        "../tricked_ops.so",
        "../../tricked_ops.so",
        "../../../tricked_ops.so",
        "./scripts/tricked_ops.so",
    ];
    for path in lib_paths {
        if let Ok(lib) = Library::new(path) {
            return Some(lib);
        }
    }
    None
}
