from datetime import timedelta
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from stations.models import Organization, OrganizationInvitation, OrganizationMember, Station


class StationAccessTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner", email="owner@example.com", password="pass12345")
        self.other = User.objects.create_user(username="other", email="other@example.com", password="pass12345")
        self.org = Organization.objects.create(name="Org A", owner=self.owner)
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )

    def test_member_can_create_station_in_own_org(self):
        self.client.force_authenticate(self.owner)

        response = self.client.post(
            "/api/stations/",
            {"name": "Station A", "org": self.org.id},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Station.objects.filter(org=self.org, name="Station A").exists())

    def test_non_member_cannot_create_station_in_foreign_org(self):
        self.client.force_authenticate(self.other)

        response = self.client.post(
            "/api/stations/",
            {"name": "Station A", "org": self.org.id},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("org", response.data)
        self.assertFalse(Station.objects.filter(org=self.org, name="Station A").exists())


class StationAutoHistoryFolderTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="folder-owner", email="owner-folder@example.com", password="pass12345")
        self.org = Organization.objects.create(name="Folder Org", owner=self.owner)
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )

    def test_new_station_gets_dedicated_auto_history_folder_from_name(self):
        station = Station.objects.create(
            org=self.org,
            name="SES 1.2 MW",
            capacity_mw=1.2,
        )

        self.assertEqual(station.auto_history_folder, f"/mnt/share/org_{self.org.id}/SES_1.2_MW")


    def test_default_auto_history_folder_is_created_on_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / f"org_{self.org.id}" / "SES_1.2_MW"
            with patch.object(Station, "_build_auto_history_folder", return_value=str(expected)):
                station = Station.objects.create(
                    org=self.org,
                    name="SES 1.2 MW",
                    capacity_mw=1.2,
                    auto_history_folder="/mnt/share",
                )

            self.assertEqual(station.auto_history_folder, str(expected))
            self.assertTrue(expected.exists())

    def test_custom_auto_history_folder_is_preserved(self):
        station = Station.objects.create(
            org=self.org,
            name="SES 8.8 MW",
            capacity_mw=8.8,
            auto_history_folder="/mnt/share/custom-folder",
        )

        self.assertEqual(station.auto_history_folder, "/mnt/share/custom-folder")

    def test_same_station_name_in_different_orgs_gets_different_folders(self):
        other_owner = User.objects.create_user(username="folder-owner-2", email="owner-folder-2@example.com", password="pass12345")
        other_org = Organization.objects.create(name="Other Folder Org", owner=other_owner)

        station_a = Station.objects.create(org=self.org, name="SES 8.8 MW", capacity_mw=8.8)
        station_b = Station.objects.create(org=other_org, name="SES 8.8 MW", capacity_mw=8.8)

        self.assertEqual(station_a.auto_history_folder, f"/mnt/share/org_{self.org.id}/SES_8.8_MW")
        self.assertEqual(station_b.auto_history_folder, f"/mnt/share/org_{other_org.id}/SES_8.8_MW")
        self.assertNotEqual(station_a.auto_history_folder, station_b.auto_history_folder)


    def test_management_command_creates_folder_for_existing_station(self):
        station = Station.objects.create(
            org=self.org,
            name="SES Existing",
            capacity_mw=1.2,
            auto_history_folder="/mnt/share/custom-folder",
        )

        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / f"org_{self.org.id}" / "SES_Existing"
            Station.objects.filter(pk=station.pk).update(auto_history_folder=str(expected))
            self.assertFalse(expected.exists())

            call_command("ensure_station_import_folders", "--station-id", str(station.pk))

            self.assertTrue(expected.exists())

    def test_existing_station_with_default_folder_gets_dedicated_folder_on_save(self):
        station = Station.objects.create(
            org=self.org,
            name="SES 8.8 MW",
            capacity_mw=8.8,
            auto_history_folder="/mnt/share/custom-folder",
        )

        station.auto_history_folder = "/mnt/share"
        station.save(update_fields=["auto_history_folder"])
        station.refresh_from_db()

        self.assertEqual(station.auto_history_folder, f"/mnt/share/org_{self.org.id}/SES_8.8_MW")


class InvitationFlowTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner", email="owner@example.com", password="pass12345")
        self.analyst = User.objects.create_user(username="analyst", email="analyst@example.com", password="pass12345")
        self.invited = User.objects.create_user(username="invitee", email="invitee@example.com", password="pass12345")

        self.org = Organization.objects.create(name="Org A", owner=self.owner)
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.analyst,
            role=OrganizationMember.ROLE_ANALYST,
        )

    def test_owner_can_create_invitation_but_analyst_cannot(self):
        self.client.force_authenticate(self.owner)
        ok = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "new.user@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(ok.status_code, status.HTTP_201_CREATED)

        self.client.force_authenticate(self.analyst)
        forbidden = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "another.user@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(forbidden.status_code, status.HTTP_403_FORBIDDEN)

    def test_accept_invitation_requires_matching_email(self):
        invitation = OrganizationInvitation.objects.create(
            organization=self.org,
            invited_email="invitee@example.com",
            role=OrganizationMember.ROLE_ADMIN,
            invited_by=self.owner,
            expires_at=timezone.now() + timedelta(days=1),
        )

        self.client.force_authenticate(self.analyst)
        bad = self.client.post(
            "/api/orgs/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(bad.status_code, status.HTTP_400_BAD_REQUEST)

        self.client.force_authenticate(self.invited)
        ok = self.client.post(
            "/api/orgs/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(ok.status_code, status.HTTP_200_OK)

        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.STATUS_ACCEPTED)
        self.assertTrue(
            OrganizationMember.objects.filter(
                organization=self.org,
                user=self.invited,
                role=OrganizationMember.ROLE_ADMIN,
            ).exists()
        )


class TrialEnforcementTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="trialowner", email="trialowner@example.com", password="pass12345")
        self.org = Organization.objects.create(
            name="Expired Org",
            owner=self.owner,
            trial_ends_at=timezone.now() - timedelta(days=1),
            is_active=True,
        )
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )

    def test_station_create_blocked_when_trial_expired(self):
        self.client.force_authenticate(self.owner)
        response = self.client.post(
            "/api/stations/",
            {"name": "Blocked Station", "org": self.org.id},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertFalse(Station.objects.filter(org=self.org, name="Blocked Station").exists())

    def test_invitation_create_blocked_when_trial_expired(self):
        self.client.force_authenticate(self.owner)
        response = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "new.user@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertFalse(OrganizationInvitation.objects.filter(organization=self.org).exists())


class InvitationManagementTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner2", email="owner2@example.com", password="pass12345")
        self.org = Organization.objects.create(name="Org B", owner=self.owner, trial_ends_at=timezone.now() + timedelta(days=1))
        OrganizationMember.objects.create(organization=self.org, user=self.owner, role=OrganizationMember.ROLE_OWNER)

    @patch("stations.views.send_mail")
    def test_invitation_create_sends_email(self, send_mail_mock):
        self.client.force_authenticate(self.owner)
        response = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "notify@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        send_mail_mock.assert_called_once()

    def test_invitation_throttle_blocks_after_limit(self):
        self.client.force_authenticate(self.owner)
        for i in range(10):
            OrganizationInvitation.objects.create(
                organization=self.org,
                invited_email=f"u{i}@example.com",
                role=OrganizationMember.ROLE_VIEWER,
                invited_by=self.owner,
                expires_at=timezone.now() + timedelta(days=2),
            )

        response = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "blocked@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch("stations.views.send_mail")
    def test_owner_can_resend_and_revoke_invitation(self, send_mail_mock):
        invitation = OrganizationInvitation.objects.create(
            organization=self.org,
            invited_email="resend@example.com",
            role=OrganizationMember.ROLE_VIEWER,
            invited_by=self.owner,
            expires_at=timezone.now() + timedelta(days=1),
        )
        self.client.force_authenticate(self.owner)

        resend = self.client.post(f"/api/orgs/{self.org.id}/invitations/{invitation.id}/resend/", {}, format="json")
        self.assertEqual(resend.status_code, status.HTTP_200_OK)
        self.assertEqual(resend.data["status"], "resent")
        send_mail_mock.assert_called_once()

        revoke = self.client.post(f"/api/orgs/{self.org.id}/invitations/{invitation.id}/revoke/", {}, format="json")
        self.assertEqual(revoke.status_code, status.HTTP_200_OK)
        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.STATUS_CANCELLED)


class BillingWriteAccessTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="billing", email="billing@example.com", password="pass12345")
        self.org = Organization.objects.create(
            name="Billing Org",
            owner=self.owner,
            trial_ends_at=timezone.now() + timedelta(days=3),
            subscription_status=Organization.SUBSCRIPTION_PAST_DUE,
        )
        OrganizationMember.objects.create(organization=self.org, user=self.owner, role=OrganizationMember.ROLE_OWNER)

    def test_station_create_blocked_when_subscription_past_due(self):
        self.client.force_authenticate(self.owner)
        response = self.client.post("/api/stations/", {"name": "Past Due", "org": self.org.id}, format="json")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


class OrganizationBillingModelTests(APITestCase):
    def test_can_write_rules_for_subscription_statuses(self):
        owner = User.objects.create_user(username="owner3", email="owner3@example.com", password="pass12345")
        base_kwargs = {"name": "Rules Org", "owner": owner, "is_active": True}

        trial_active = Organization.objects.create(
            **base_kwargs,
            name="Trial active",
            subscription_status=Organization.SUBSCRIPTION_TRIALING,
            trial_ends_at=timezone.now() + timedelta(days=1),
        )
        self.assertTrue(trial_active.can_write())

        trial_expired = Organization.objects.create(
            **base_kwargs,
            name="Trial expired",
            subscription_status=Organization.SUBSCRIPTION_TRIALING,
            trial_ends_at=timezone.now() - timedelta(days=1),
        )
        self.assertFalse(trial_expired.can_write())

        active = Organization.objects.create(
            **base_kwargs,
            name="Active",
            subscription_status=Organization.SUBSCRIPTION_ACTIVE,
            trial_ends_at=timezone.now() - timedelta(days=100),
        )
        self.assertTrue(active.can_write())

        past_due = Organization.objects.create(
            **base_kwargs,
            name="Past due",
            subscription_status=Organization.SUBSCRIPTION_PAST_DUE,
            trial_ends_at=timezone.now() + timedelta(days=1),
        )
        self.assertFalse(past_due.can_write())
