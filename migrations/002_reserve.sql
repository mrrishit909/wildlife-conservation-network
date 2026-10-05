-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- Positions are x / y kilometres on the synthetic reserve grid, not PostGIS geography.

CREATE TABLE reserve (tenant_id uuid PRIMARY KEY REFERENCES tenants, started_at timestamptz NOT NULL, seed int NOT NULL);       -- (+) day 0 of the month on record
SELECT enable_tenant_rls('reserve');

CREATE TABLE species (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, UNIQUE (tenant_id, name));   -- (+)
SELECT enable_tenant_rls('species');

CREATE TABLE sensors (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, sensor_type text NOT NULL CHECK (sensor_type IN ('camera','acoustic')), x real NOT NULL, y real NOT NULL, deployed_at timestamptz NOT NULL,
  status text NOT NULL DEFAULT 'active', name text NOT NULL, ordinal int NOT NULL, last_seq int NOT NULL DEFAULT -1);                 -- (+)
SELECT enable_tenant_rls('sensors');

-- (+) Sensors work offline and upload numbered batches when they can. A batch is stored once, whatever order it arrives in.
CREATE TABLE sync_batches (sensor_id uuid NOT NULL REFERENCES sensors ON DELETE CASCADE, seq int NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, received_at timestamptz NOT NULL DEFAULT now(), observations int NOT NULL,
  arrived_out_of_order boolean NOT NULL, PRIMARY KEY (sensor_id, seq));
SELECT enable_tenant_rls('sync_batches');

CREATE TABLE observations (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, sensor_id uuid NOT NULL REFERENCES sensors ON DELETE CASCADE, observed_at timestamptz NOT NULL, species_id uuid REFERENCES species,
  confidence real, count int NOT NULL DEFAULT 1, media_uri text, x real NOT NULL, y real NOT NULL,
  kind text NOT NULL CHECK (kind IN ('animal','vehicle','sound')), label text, day int NOT NULL, hour real NOT NULL, batch_seq int NOT NULL, features jsonb, sound_label text, sound_p real);   -- (+)
CREATE INDEX observations_sensor ON observations (sensor_id, day);
CREATE INDEX observations_kind ON observations (tenant_id, kind, label);
SELECT enable_tenant_rls('observations');

CREATE TABLE animals (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, species_id uuid NOT NULL REFERENCES species, tag_id text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}'::jsonb, UNIQUE (tenant_id, tag_id));
SELECT enable_tenant_rls('animals');

CREATE TABLE tracks (animal_id uuid REFERENCES animals ON DELETE CASCADE, recorded_at timestamptz, x real NOT NULL, y real NOT NULL, source text NOT NULL, PRIMARY KEY (animal_id, recorded_at),
  tenant_id uuid NOT NULL REFERENCES tenants, day int NOT NULL);                                                                        -- (+)
SELECT enable_tenant_rls('tracks');

CREATE TABLE patrol_events (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, ranger_team text NOT NULL, event_type text NOT NULL, x real NOT NULL, y real NOT NULL, occurred_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb);
SELECT enable_tenant_rls('patrol_events');
