export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      attempts: {
        Row: {
          confidence: number | null
          created_at: string
          hint_shown: string | null
          hint_source: Database["public"]["Enums"]["hint_source"]
          id: string
          learner_disagreed: boolean
          mediapipe_detection_failed: boolean
          model_version_id: string | null
          passed: boolean
          predicted_class_id: string | null
          prompted_at: string
          submitted_at: string
          user_id: string
          vocab_id: string
        }
        Insert: {
          confidence?: number | null
          created_at?: string
          hint_shown?: string | null
          hint_source?: Database["public"]["Enums"]["hint_source"]
          id?: string
          learner_disagreed?: boolean
          mediapipe_detection_failed?: boolean
          model_version_id?: string | null
          passed: boolean
          predicted_class_id?: string | null
          prompted_at: string
          submitted_at: string
          user_id: string
          vocab_id: string
        }
        Update: {
          confidence?: number | null
          created_at?: string
          hint_shown?: string | null
          hint_source?: Database["public"]["Enums"]["hint_source"]
          id?: string
          learner_disagreed?: boolean
          mediapipe_detection_failed?: boolean
          model_version_id?: string | null
          passed?: boolean
          predicted_class_id?: string | null
          prompted_at?: string
          submitted_at?: string
          user_id?: string
          vocab_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "attempts_model_version_id_fkey"
            columns: ["model_version_id"]
            isOneToOne: false
            referencedRelation: "model_versions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "attempts_predicted_class_id_fkey"
            columns: ["predicted_class_id"]
            isOneToOne: false
            referencedRelation: "vocabulary_items"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "attempts_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "attempts_vocab_id_fkey"
            columns: ["vocab_id"]
            isOneToOne: false
            referencedRelation: "vocabulary_items"
            referencedColumns: ["id"]
          },
        ]
      }
      confusion_pair_hints: {
        Row: {
          created_at: string
          hint: string
          id: string
          predicted_sign_id: string
          target_sign_id: string
        }
        Insert: {
          created_at?: string
          hint: string
          id?: string
          predicted_sign_id: string
          target_sign_id: string
        }
        Update: {
          created_at?: string
          hint?: string
          id?: string
          predicted_sign_id?: string
          target_sign_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "confusion_pair_hints_predicted_sign_id_fkey"
            columns: ["predicted_sign_id"]
            isOneToOne: false
            referencedRelation: "vocabulary_items"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "confusion_pair_hints_target_sign_id_fkey"
            columns: ["target_sign_id"]
            isOneToOne: false
            referencedRelation: "vocabulary_items"
            referencedColumns: ["id"]
          },
        ]
      }
      mastery_state: {
        Row: {
          consecutive_passes: number
          ease: number
          interval_days: number
          last_attempt_at: string | null
          next_review_at: string | null
          status: Database["public"]["Enums"]["mastery_status"]
          total_attempts: number
          total_passes: number
          updated_at: string
          user_id: string
          vocab_id: string
        }
        Insert: {
          consecutive_passes?: number
          ease?: number
          interval_days?: number
          last_attempt_at?: string | null
          next_review_at?: string | null
          status?: Database["public"]["Enums"]["mastery_status"]
          total_attempts?: number
          total_passes?: number
          updated_at?: string
          user_id: string
          vocab_id: string
        }
        Update: {
          consecutive_passes?: number
          ease?: number
          interval_days?: number
          last_attempt_at?: string | null
          next_review_at?: string | null
          status?: Database["public"]["Enums"]["mastery_status"]
          total_attempts?: number
          total_passes?: number
          updated_at?: string
          user_id?: string
          vocab_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "mastery_state_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "mastery_state_vocab_id_fkey"
            columns: ["vocab_id"]
            isOneToOne: false
            referencedRelation: "vocabulary_items"
            referencedColumns: ["id"]
          },
        ]
      }
      model_versions: {
        Row: {
          artifact_url: string
          config_url: string
          created_at: string
          id: string
          is_active: boolean
          metrics_url: string
          promoted_at: string | null
          promoted_by: string | null
        }
        Insert: {
          artifact_url: string
          config_url: string
          created_at?: string
          id: string
          is_active?: boolean
          metrics_url: string
          promoted_at?: string | null
          promoted_by?: string | null
        }
        Update: {
          artifact_url?: string
          config_url?: string
          created_at?: string
          id?: string
          is_active?: boolean
          metrics_url?: string
          promoted_at?: string | null
          promoted_by?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "model_versions_promoted_by_fkey"
            columns: ["promoted_by"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      users: {
        Row: {
          created_at: string
          email: string | null
          fitzpatrick: number | null
          handedness: Database["public"]["Enums"]["handedness"]
          id: string
          onboarded_at: string | null
          updated_at: string
        }
        Insert: {
          created_at?: string
          email?: string | null
          fitzpatrick?: number | null
          handedness?: Database["public"]["Enums"]["handedness"]
          id: string
          onboarded_at?: string | null
          updated_at?: string
        }
        Update: {
          created_at?: string
          email?: string | null
          fitzpatrick?: number | null
          handedness?: Database["public"]["Enums"]["handedness"]
          id?: string
          onboarded_at?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      vocabulary_items: {
        Row: {
          asl_lex_code: string | null
          asl_lex_match: string | null
          category: string
          created_at: string
          difficulty_rank: number | null
          display_gloss: string
          flippable: boolean
          generic_failure_hint: string | null
          id: string
          lifeprint_lesson: string | null
          parameters: Json | null
          pre_attempt_hint: string | null
          reference_video_url: string | null
          static_or_movement: Database["public"]["Enums"]["sign_movement"]
          wlasl_clip_count: number | null
        }
        Insert: {
          asl_lex_code?: string | null
          asl_lex_match?: string | null
          category: string
          created_at?: string
          difficulty_rank?: number | null
          display_gloss: string
          flippable: boolean
          generic_failure_hint?: string | null
          id: string
          lifeprint_lesson?: string | null
          parameters?: Json | null
          pre_attempt_hint?: string | null
          reference_video_url?: string | null
          static_or_movement: Database["public"]["Enums"]["sign_movement"]
          wlasl_clip_count?: number | null
        }
        Update: {
          asl_lex_code?: string | null
          asl_lex_match?: string | null
          category?: string
          created_at?: string
          difficulty_rank?: number | null
          display_gloss?: string
          flippable?: boolean
          generic_failure_hint?: string | null
          id?: string
          lifeprint_lesson?: string | null
          parameters?: Json | null
          pre_attempt_hint?: string | null
          reference_video_url?: string | null
          static_or_movement?: Database["public"]["Enums"]["sign_movement"]
          wlasl_clip_count?: number | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      cleanup_inactive_anonymous_users: { Args: never; Returns: number }
    }
    Enums: {
      handedness: "right" | "left" | "ambidextrous" | "unspecified"
      hint_source: "confusion_pair" | "generic_failure" | "none"
      mastery_status: "untouched" | "learning" | "reviewing" | "mastered"
      sign_movement: "static" | "movement"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      handedness: ["right", "left", "ambidextrous", "unspecified"],
      hint_source: ["confusion_pair", "generic_failure", "none"],
      mastery_status: ["untouched", "learning", "reviewing", "mastered"],
      sign_movement: ["static", "movement"],
    },
  },
} as const
