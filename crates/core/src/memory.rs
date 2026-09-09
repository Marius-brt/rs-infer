//! Lightweight process memory stats (no heavy dependencies).

/// Current memory footprint of this process, in bytes, if obtainable.
///
/// On macOS this is `phys_footprint` (what Activity Monitor / `top` report, excluding
/// purgeable and mmap'd file pages); on Linux it is RSS from `/proc/self/statm`.
pub fn rss_bytes() -> Option<u64> {
	#[cfg(target_os = "linux")]
	{
		let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
		let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
		let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
		(page > 0).then(|| resident_pages * page as u64)
	}
	#[cfg(target_os = "macos")]
	{
		// task_info(TASK_VM_INFO) exposes phys_footprint, the metric Activity Monitor / `top` report.
		// The trailing pad keeps the buffer big enough for newer kernel revisions of the struct;
		// an exact-size mismatch just yields an error and we fall back to mach_task_basic_info.
		#[repr(C)]
		struct TaskVmInfo {
			virtual_size: u64,
			region_count: libc::c_int,
			page_size: libc::c_int,
			resident_size: u64,
			resident_size_peak: u64,
			device: u64,
			device_peak: u64,
			internal: u64,
			internal_peak: u64,
			external: u64,
			external_peak: u64,
			reusable: u64,
			reusable_peak: u64,
			purgeable_volatile_pmap: u64,
			purgeable_volatile_resident: u64,
			purgeable_volatile_virtual: u64,
			compressed: u64,
			compressed_peak: u64,
			compressed_lifetime: u64,
			phys_footprint: u64,
			phys_footprint_peak: u64,
			_pad: [u8; 160],
		}
		const TASK_VM_INFO: libc::c_uint = 22;
		let mut info: TaskVmInfo = unsafe { std::mem::zeroed() };
		let mut count = (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<libc::c_int>()) as libc::c_uint;
		#[allow(deprecated)]
		let kr = unsafe {
			libc::task_info(
				libc::mach_task_self(),
				TASK_VM_INFO,
				&mut info as *mut TaskVmInfo as libc::task_info_t,
				&mut count,
			)
		};
		// KERN_SUCCESS == 0
		if kr == 0 && info.phys_footprint > 0 {
			return Some(info.phys_footprint);
		}
		#[allow(deprecated)]
		unsafe {
			let mut basic: libc::mach_task_basic_info = std::mem::zeroed();
			let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
			let kr = libc::task_info(
				libc::mach_task_self(),
				libc::MACH_TASK_BASIC_INFO,
				&mut basic as *mut libc::mach_task_basic_info as libc::task_info_t,
				&mut count,
			);
			// KERN_SUCCESS == 0
			(kr == 0).then_some(basic.resident_size as u64)
		}
	}
	#[cfg(not(any(target_os = "linux", target_os = "macos")))]
	{
		None
	}
}

/// Current memory footprint in mebibytes, or None if unavailable on this platform.
pub fn rss_mb() -> Option<u64> {
	rss_bytes().map(|b| b / (1024 * 1024))
}

/// Size of an ONNX model on disk in mebibytes, counting the graph file plus any
/// external-data siblings (e.g. `model.onnx` + `model.onnx.data`).
pub fn model_size_mb(model_file: &std::path::Path) -> Option<u64> {
	let mut total = std::fs::metadata(model_file).ok()?.len();
	let name = model_file.file_name().and_then(|s| s.to_str())?;
	if let Some(dir) = model_file.parent() {
		if let Ok(entries) = std::fs::read_dir(dir) {
			for entry in entries.flatten() {
				let path = entry.path();
				if path == model_file {
					continue;
				}
				let sibling = path.file_name().and_then(|s| s.to_str());
				// External data files are named "<model>.onnx_data" (or _1, _2, ...).
				if sibling.is_some_and(|s| s.starts_with(name) && s.len() > name.len()) {
					if let Ok(m) = entry.metadata() {
						if m.is_file() {
							total += m.len();
						}
					}
				}
			}
		}
	}
	Some(total / (1024 * 1024))
}
