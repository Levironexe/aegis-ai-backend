#!/usr/bin/env python3
"""
Migration script to add provider column to Message_v2 table
"""
import asyncio
import asyncpg
from app.config import settings

async def run_migration():
    # Parse the database URL
    db_url = settings.database_url
    # Convert asyncpg URL format
    db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')

    print(f"Connecting to database...")

    try:
        # Connect to the database
        conn = await asyncpg.connect(db_url)

        # Check if column already exists
        check_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'Message_v2'
        AND column_name = 'provider';
        """

        result = await conn.fetch(check_query)

        if result:
            print("✓ Column 'provider' already exists in Message_v2 table")
        else:
            print("Adding 'provider' column to Message_v2 table...")

            # Add the provider column
            await conn.execute('''
                ALTER TABLE "Message_v2"
                ADD COLUMN provider VARCHAR(20) NULL;
            ''')

            print("✓ Successfully added 'provider' column to Message_v2 table")

        # Verify the column was added
        verify_query = """
        SELECT column_name, data_type, character_maximum_length, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'Message_v2'
        AND column_name = 'provider';
        """

        result = await conn.fetch(verify_query)
        if result:
            row = result[0]
            print(f"\nColumn details:")
            print(f"  Name: {row['column_name']}")
            print(f"  Type: {row['data_type']}({row['character_maximum_length']})")
            print(f"  Nullable: {row['is_nullable']}")

        await conn.close()
        print("\n✓ Migration completed successfully!")

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
