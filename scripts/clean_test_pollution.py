import sqlite3
import shutil
from pathlib import Path

def clean_db(db_path: Path):
    if not db_path.exists():
        return
    print(f"Cleaning test pollution in {db_path}...")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Beings to remove: any tenant_id starting with 'test-' or 'tenant-'
    beings_to_remove = cur.execute(
        "SELECT being_id, tenant_id FROM beings WHERE tenant_id LIKE 'test-%' OR tenant_id LIKE 'tenant-%'"
    ).fetchall()
    
    for being_id, tenant_id in beings_to_remove:
        print(f"  Removing being {being_id} ({tenant_id})")
        cur.execute("DELETE FROM being_minds WHERE being_id = ?", (being_id,))
        cur.execute("DELETE FROM beings WHERE being_id = ?", (being_id,))
        
    con.commit()
    con.close()

def clean_craft(craft_dir: Path):
    if not craft_dir.exists():
        return
    print(f"Cleaning test craft in {craft_dir}...")
    for item in craft_dir.iterdir():
        if item.name.startswith("engagement-") or item.name.startswith("mission-msn-"):
            print(f"  Removing test craft dir {item.name}")
            shutil.rmtree(item, ignore_errors=True)

if __name__ == "__main__":
    clean_db(Path("sonic-core/sonic_data.db"))
    clean_db(Path("sonic_data.db"))
    clean_craft(Path("sonic-core/sonic_data/craft"))
    clean_craft(Path("sonic_data/craft"))
    print("Test pollution cleanup completed.")

