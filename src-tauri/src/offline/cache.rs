//! File-backed cache for provider health, inventories, and optional payloads.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::offline::storage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub stored_at: u64,
    pub ttl_secs: u64,
    pub value: T,
}

#[derive(Debug, Clone)]
pub struct FileCache {
    root: PathBuf,
}

impl FileCache {
    pub fn new() -> Self {
        Self {
            root: storage::cache_dir(),
        }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self { root }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        self.root.join(format!("{safe}.json"))
    }

    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        let path = self.path_for(key);
        let raw = fs::read_to_string(path).ok()?;
        let entry: CacheEntry<T> = serde_json::from_str(&raw).ok()?;
        let now = now_secs();
        if entry.ttl_secs > 0 && now.saturating_sub(entry.stored_at) > entry.ttl_secs {
            return None;
        }
        Some(entry.value)
    }

    pub fn set<T: Serialize>(&self, key: &str, value: &T, ttl_secs: u64) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        let entry = CacheEntry {
            stored_at: now_secs(),
            ttl_secs,
            value,
        };
        let json = serde_json::to_string_pretty(&entry).map_err(|e| e.to_string())?;
        fs::write(self.path_for(key), json).map_err(|e| e.to_string())
    }

    pub fn remove(&self, key: &str) -> Result<(), String> {
        let path = self.path_for(key);
        if path.exists() {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn clear(&self) -> Result<(), String> {
        if self.root.exists() {
            fs::remove_dir_all(&self.root).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())
    }
}

impl Default for FileCache {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn set_get_roundtrip() {
        let dir = tempdir().unwrap();
        let cache = FileCache::with_root(dir.path().to_path_buf());
        cache.set("hello", &"world".to_string(), 60).unwrap();
        let value: String = cache.get("hello").expect("cached");
        assert_eq!(value, "world");
    }

    #[test]
    fn expired_entry_is_none() {
        let dir = tempdir().unwrap();
        let cache = FileCache::with_root(dir.path().to_path_buf());
        let entry = CacheEntry {
            stored_at: 1,
            ttl_secs: 1,
            value: 42u32,
        };
        let path = cache.path_for("old");
        fs::write(path, serde_json::to_string(&entry).unwrap()).unwrap();
        let value: Option<u32> = cache.get("old");
        assert!(value.is_none());
    }

    #[test]
    fn clear_removes_files() {
        let dir = tempdir().unwrap();
        let cache = FileCache::with_root(dir.path().to_path_buf());
        cache.set("a", &1u32, 0).unwrap();
        cache.clear().unwrap();
        let value: Option<u32> = cache.get("a");
        assert!(value.is_none());
    }
}
